"""DiT feature cache (TeaCache-style) for the WAN diffusion transformer.

Net-new (no equivalent in the upstream tree). Opt-in via
``enable_dit_cache(model)``. Across diffusion steps the block-stack output changes
slowly; we cache the residual ``(x_after_blocks - x_before_blocks)`` and REUSE it on
steps where the timestep-modulation ``e0`` (and thus the denoising direction) barely
changed, skipping the expensive block loop. Resets automatically at each new generation
(detected by the timestep jumping back up). See docs/SERVER_PROTOCOL_SPEC.md speed notes.
"""
from __future__ import annotations

import math

import torch
from einops import repeat
from torch.utils.checkpoint import checkpoint
from functools import partial

from .modules.model import sinusoidal_embedding_1d


def enable_dit_cache(model, rel_l1_thresh: float = 0.15) -> None:
    """Monkeypatch ``model.forward`` (a WanModel) with the cached variant.

    rel_l1_thresh: accumulate the per-step relative-L1 change of e0; recompute the
    block loop only once the accumulation crosses this threshold (else reuse cache).
    Higher threshold = more skips = faster but more quality drift. 0.15 is a moderate default.
    """
    if getattr(model, "_dit_cache_on", False):
        return
    model._dit_cache_on = True
    model._tc = {"prev_e0": None, "prev_t": None, "acc": 0.0, "residual": None,
                 "thresh": float(rel_l1_thresh), "n_calc": 0, "n_skip": 0}

    def cached_forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        n_frames = x.shape[2]
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings (mirror WanModel.forward)
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        t_shape = tuple(t.shape)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        if t.ndim == 2:
            e = e.unflatten(dim=0, sizes=t_shape)
        else:
            e = repeat(e, "b c -> b f c", f=n_frames)
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))

        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )
        if clip_fea is not None:
            context = torch.concat([self.img_emb(clip_fea), context], dim=1)

        # ---- TeaCache decision ------------------------------------------------
        tc = self._tc
        cur_t = float(t.float().mean())
        # new generation? (denoising t decreases within a run; a jump up = new run)
        if tc["prev_t"] is None or cur_t > tc["prev_t"] + 1e-3:
            tc["prev_e0"] = None; tc["acc"] = 0.0; tc["residual"] = None
        tc["prev_t"] = cur_t

        if tc["prev_e0"] is not None and tc["prev_e0"].shape != e0.shape:
            # context length changed between generates (e.g. ctx 19 -> 21 as the client's
            # rolling window fills): prev_e0's latent-frame count no longer matches. The
            # timestep-jump heuristic above can miss this — treat as a fresh generation.
            tc["prev_e0"] = None; tc["acc"] = 0.0; tc["residual"] = None
        if tc["prev_e0"] is None:
            should_calc = True
        else:
            rel = ((e0 - tc["prev_e0"]).abs().mean() / (tc["prev_e0"].abs().mean() + 1e-8)).item()
            tc["acc"] += rel
            should_calc = tc["acc"] >= tc["thresh"]
            if should_calc:
                tc["acc"] = 0.0
        tc["prev_e0"] = e0.detach()

        if should_calc or tc["residual"] is None:
            tc["n_calc"] += 1
            x_in = x
            x = x.clone()
            kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                          freqs=self.freqs, context=context, context_lens=context_lens)
            for i, block in enumerate(self.blocks):
                blk = partial(block, **kwargs)
                if i in self.gradient_checkpointing_indices:
                    x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
            tc["residual"] = (x - x_in).detach()
        else:
            tc["n_skip"] += 1
            x = x + tc["residual"]

        # head + unpatchify
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    import types
    model.forward = types.MethodType(cached_forward, model)


def cache_stats(model) -> dict:
    tc = getattr(model, "_tc", None)
    if not tc:
        return {}
    total = tc["n_calc"] + tc["n_skip"]
    return {"calc": tc["n_calc"], "skip": tc["n_skip"],
            "skip_frac": tc["n_skip"] / total if total else 0.0}
