import logging
import gc
import pickle
import torch
import numpy as np
import torch.distributed as dist
from einops import rearrange, repeat
from tqdm import tqdm
from vera.video_model.algorithms.common.base_pytorch_algo import BasePytorchAlgo
from transformers import get_scheduler
import zmq
import msgpack
import io
from PIL import Image
import torchvision.transforms as transforms
from vera.video_model.utils.video_utils import numpy_to_mp4_bytes

from .modules.model import WanModel, WanAttentionBlock
from .modules.t5 import umt5_xxl, T5CrossAttention, T5SelfAttention, T5Encoder
from .modules.tokenizers import HuggingfaceTokenizer
from .modules.vae import video_vae_factory
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.vis_utils import (
    add_red_border,
    pad_temporal,
    build_pred_flow_panel,
    build_gt_flow_panel,
    flow_to_rgb,
)
from vera.video_model.utils.ckpt_utils import extract_flow_decoder_vae_state_dict
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata
from omegaconf.listconfig import ListConfig


def _load_checkpoint_weights_only(path, *, map_location="cpu", mmap=True):
    # PyTorch 2.6 defaults torch.load(..., weights_only=True), but WAN training
    # checkpoints can serialize OmegaConf config objects alongside the state dict.
    torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata])
    try:
        return torch.load(
            path,
            mmap=mmap,
            map_location=map_location,
            weights_only=True,
        )
    except pickle.UnpicklingError:
        logging.warning(
            "Falling back to torch.load(..., weights_only=False) for trusted WAN "
            "checkpoint %s because it contains non-tensor Python objects.",
            path,
        )
        return torch.load(
            path,
            mmap=mmap,
            map_location=map_location,
            weights_only=False,
        )


def print_module_hierarchy(model, indent=0):
    for name, module in model.named_children():
        print(" " * indent + f"{name}: {type(module)}")
        print_module_hierarchy(module, indent + 2)


class WanTextToVideo(BasePytorchAlgo):
    """
    Main class for WanTextToVideo
    """

    def __init__(self, cfg):
        self.num_train_timesteps = cfg.num_train_timesteps
        self.height = cfg.height
        self.width = cfg.width
        self.n_frames = cfg.n_frames
        self.gradient_checkpointing_rate = cfg.gradient_checkpointing_rate
        self.sample_solver = cfg.sample_solver
        self.sample_steps = cfg.sample_steps
        self.sample_shift = cfg.sample_shift
        self.lang_guidance = cfg.lang_guidance
        self.neg_prompt = cfg.neg_prompt
        self.hist_guidance = cfg.hist_guidance
        self.sliding_hist = cfg.sliding_hist
        self.diffusion_forcing = cfg.diffusion_forcing
        self.vae_stride = cfg.vae.stride
        self.patch_size = cfg.model.patch_size
        self.diffusion_type = cfg.diffusion_type  # "discrete"  # or "continuous"
        self.M = cfg.diffusion_forcing.get("M", None)
        self.N = cfg.diffusion_forcing.get("N", None)
        self.skip_text_encoder = cfg.get("skip_text_encoder", False)
        self.vae_num_views = int(cfg.get("vae_num_views", 1))
        self.flow_decoder_cfg = cfg.get("flow_decoder", None)
        self.flow_train_multiplier = (
            self.flow_decoder_cfg.get("flow_train_multiplier", 1.0)
            if self.flow_decoder_cfg is not None
            else 1.0
        )
        inference_cfg = cfg.get("inference", {})
        decode_outputs = inference_cfg.get("decode_outputs", ["rgb"])
        if decode_outputs is None:
            decode_outputs = ["rgb"]
        elif isinstance(decode_outputs, str):
            decode_outputs = [decode_outputs]
        else:
            decode_outputs = list(decode_outputs)
        self.decode_outputs_cfg = decode_outputs
        validation_n_frames = cfg.diffusion_forcing.get("validation_n_frames", None)
        self.validation_lat_t = (
            1 + (validation_n_frames - 1) // cfg.vae.stride[0]
            if validation_n_frames is not None
            else None
        )

        self.lat_h = self.height // self.vae_stride[1]
        self.lat_w = self.width // self.vae_stride[2]
        self.lat_t = 1 + (self.n_frames - 1) // self.vae_stride[0]
        self.lat_c = cfg.vae.z_dim
        self.max_area = self.height * self.width
        self.max_tokens = (
            self.lat_t
            * self.lat_h
            * self.lat_w
            // (self.patch_size[1] * self.patch_size[2])
        )

        self.load_prompt_embed = cfg.load_prompt_embed
        self.load_video_latent = cfg.load_video_latent
        self.socket = None
        self._loss_buffer = []
        self._loss_log_interval = cfg.logging.get("loss_smooth_interval", 10)
        if (self.sliding_hist - 1) % self.vae_stride[0] != 0:
            raise ValueError(
                "sliding_hist - 1 must be a multiple of vae_stride[0] due to temporal "
                f"vae. Got {self.sliding_hist} and vae stride {self.vae_stride[0]}"
            )
        super().__init__(cfg)

    @staticmethod
    def classes_to_shard():
        # WanVAE_ excluded: Conv layers can corrupt under per-module FSDP wrapping.
        classes = {WanAttentionBlock, T5CrossAttention, T5SelfAttention, T5Encoder}
        return classes

    @property
    def device(self):
        """Device for inference; when set by FSDP adapter, use that instead of Lightning's."""
        return self.__dict__.get("device", super().device)

    @device.setter
    def device(self, value):
        self.__dict__["device"] = value

    @property
    def is_inference(self) -> bool:
        return self._trainer is None or not self.trainer.training

    @property
    def flow_decoder_vae(self):
        """Flow decoder stored outside nn.Module tree to avoid FSDP sharding."""
        if hasattr(self, "_flow_decoder_container") and self._flow_decoder_container:
            return self._flow_decoder_container[0]
        return None

    def configure_model(self):
        logging.info("Building model...")
        # Initialize text encoder
        if self.skip_text_encoder:
            self.text_encoder = None
            self.tokenizer = None
        elif not self.cfg.load_prompt_embed or self.lang_guidance:
            # T5 checkpoint is bf16; create on CPU so load_state_dict converts bf16→fp32
            text_encoder = (
                umt5_xxl(
                    encoder_only=True,
                    return_tokenizer=False,
                    dtype=torch.bfloat16 if self.is_inference else self.dtype,
                    device=torch.device("cpu"),
                )
                .eval()
                .requires_grad_(False)
            )
            if self.cfg.text_encoder.ckpt_path is not None:
                text_encoder.load_state_dict(
                    torch.load(
                        self.cfg.text_encoder.ckpt_path,
                        map_location="cpu",
                        weights_only=True,
                        mmap=True,
                    )
                )
            if self.cfg.text_encoder.compile:
                text_encoder = torch.compile(text_encoder)
            self.text_encoder = text_encoder

            # Initialize tokenizer
            self.tokenizer = HuggingfaceTokenizer(
                name=self.cfg.text_encoder.name,
                seq_len=self.cfg.text_encoder.text_len,
                clean="whitespace",
            )
        else:
            self.text_encoder = None
            self.tokenizer = HuggingfaceTokenizer(
                name=self.cfg.text_encoder.name,
                seq_len=self.cfg.text_encoder.text_len,
                clean="whitespace",
            )

        # Initialize VAE
        self.vae = (
            video_vae_factory(
                pretrained_path=self.cfg.vae.ckpt_path,
                z_dim=self.cfg.vae.z_dim,
            )
            .eval()
            .requires_grad_(False)
        ).to(self.dtype)
        self.register_buffer(
            "vae_mean", torch.tensor(self.cfg.vae.mean, dtype=self.dtype)
        )
        self.register_buffer(
            "vae_inv_std", 1.0 / torch.tensor(self.cfg.vae.std, dtype=self.dtype)
        )
        self.vae_scale = [self.vae_mean, self.vae_inv_std]
        if self.cfg.vae.compile:
            self.vae = torch.compile(self.vae)

        # Initialize main diffusion model
        if self.cfg.model.ckpt_path is None and self.cfg.model.tuned_ckpt_path is None:
            # No pretrained weights — create model from config (e.g. wan_toy)
            self.model = WanModel(
                model_type=self.cfg.model.model_type,
                patch_size=self.cfg.model.patch_size,
                text_len=self.cfg.text_encoder.text_len,
                in_dim=self.cfg.model.in_dim,
                dim=self.cfg.model.dim,
                ffn_dim=self.cfg.model.ffn_dim,
                freq_dim=self.cfg.model.freq_dim,
                text_dim=self.cfg.text_encoder.text_dim,
                out_dim=self.cfg.model.out_dim,
                num_heads=self.cfg.model.num_heads,
                num_layers=self.cfg.model.num_layers,
                window_size=self.cfg.model.window_size,
                qk_norm=self.cfg.model.qk_norm,
                cross_attn_norm=self.cfg.model.cross_attn_norm,
                eps=self.cfg.model.eps,
            )
        elif (
            getattr(self.cfg.model, "build_on_meta_for_fsdp_inference", False)
            and self.cfg.model.ckpt_path
        ):
            # FSDP inference: build on meta only; state dict loaded later by adapter.
            with torch.device("meta"):
                self.model = WanModel.from_config(
                    WanModel._dict_from_json_file(
                        self.cfg.model.ckpt_path + "/config.json"
                    )
                )
            if self.is_inference:
                self.model.to(torch.bfloat16)
            from .modules.model import rope_params

            d = self.cfg.model.dim // self.cfg.model.num_heads
            self.model.freqs = torch.cat(
                [
                    rope_params(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            )
        elif self.cfg.model.tuned_ckpt_path is None:
            self.model = WanModel.from_pretrained(self.cfg.model.ckpt_path)
        else:
            with torch.device("meta"):
                self.model = WanModel.from_config(
                    WanModel._dict_from_json_file(
                        self.cfg.model.ckpt_path + "/config.json"
                    )
                )
            if self.is_inference:
                self.model.to(torch.bfloat16)
            self.model.load_state_dict(self._load_tuned_state_dict(), assign=True)
            # Regenerate non-parameter tensors that were created on meta device
            from .modules.model import rope_params

            d = self.cfg.model.dim // self.cfg.model.num_heads
            self.model.freqs = torch.cat(
                [
                    rope_params(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            )
            # self.model = WanModel(
            #     model_type=self.cfg.model.model_type,
            #     patch_size=self.cfg.model.patch_size,
            #     text_len=self.cfg.text_encoder.text_len,
            #     in_dim=self.cfg.model.in_dim,
            #     dim=self.cfg.model.dim,
            #     ffn_dim=self.cfg.model.ffn_dim,
            #     freq_dim=self.cfg.model.freq_dim,
            #     text_dim=self.cfg.text_encoder.text_dim,
            #     out_dim=self.cfg.model.out_dim,
            #     num_heads=self.cfg.model.num_heads,
            #     num_layers=self.cfg.model.num_layers,
            #     window_size=self.cfg.model.window_size,
            #     qk_norm=self.cfg.model.qk_norm,
            #     cross_attn_norm=self.cfg.model.cross_attn_norm,
            #     eps=self.cfg.model.eps,
            # )
        if not self.is_inference:
            self.model.to(self.dtype).train()
        if self.gradient_checkpointing_rate > 0:
            self.model.gradient_checkpointing_enable(p=self.gradient_checkpointing_rate)
        if self.cfg.model.compile:
            self.model = torch.compile(self.model)

        # Log total (unsharded) model size for clarity (FSDP summary shows per-GPU)
        model_params = sum(p.numel() for p in self.model.parameters())
        ckpt_src = (
            self.cfg.model.tuned_ckpt_path or self.cfg.model.ckpt_path or "scratch"
        )
        logging.info(
            f"WanModel loaded from {ckpt_src}: "
            f"{model_params/1e9:.2f}B params (dim={self.cfg.model.dim}, "
            f"layers={self.cfg.model.num_layers}, type={self.cfg.model.model_type})"
        )

        # Initialize flow decoder (optical flow prediction from VAE latents)
        # Stored outside nn.Module tree so FSDP does not shard it.
        # This decoder is inference-only in this module; it is trained separately.
        self._flow_decoder_container = []
        if self.flow_decoder_cfg is not None and self.flow_decoder_cfg.get(
            "enabled", False
        ):
            flow_ckpt_path = self.flow_decoder_cfg.get(
                "ckpt_path", self.cfg.vae.ckpt_path
            )
            _fd = video_vae_factory(
                pretrained_path=self.cfg.vae.ckpt_path,
                z_dim=self.cfg.vae.z_dim,
                out_channels=2,
            ).to(self.dtype)
            _fd.requires_grad_(False).eval()
            # TODO: rethink this system of loading the lightning checkpoint at some point.
            if flow_ckpt_path != self.cfg.vae.ckpt_path:
                flow_state = extract_flow_decoder_vae_state_dict(flow_ckpt_path)
                model_state = _fd.state_dict()
                filtered = {
                    k: v
                    for k, v in flow_state.items()
                    if k in model_state
                    and hasattr(v, "shape")
                    and v.shape == model_state[k].shape
                }
                _fd.load_state_dict(filtered, assign=True, strict=False)
                logging.info(
                    f"Flow decoder overlay load from {flow_ckpt_path}: "
                    f"{len(filtered)}/{len(model_state)} params loaded"
                )
            self._flow_decoder_container = [_fd]
            self.register_buffer(
                "flow_vae_mean", torch.tensor(self.cfg.vae.mean, dtype=self.dtype)
            )
            self.register_buffer(
                "flow_vae_inv_std",
                1.0 / torch.tensor(self.cfg.vae.std, dtype=self.dtype),
            )
            # NOTE: same as vae_scale — do not cache in a list (stale CPU refs after .to())
            flow_params = sum(p.numel() for p in _fd.parameters())
            logging.info(
                f"Flow decoder initialized for decode-only usage: {flow_params/1e6:.1f}M params"
            )

        self.training_scheduler, self.training_timesteps = self.build_scheduler(True)

    def on_load_checkpoint(self, checkpoint):
        if self.text_encoder is None and "state_dict" in checkpoint:
            checkpoint["state_dict"] = {
                k: v
                for k, v in checkpoint["state_dict"].items()
                if not k.startswith("text_encoder.")
            }

    def configure_optimizers(self):
        # Main optimizer: WAN model + VAE (lr=0) — managed by Lightning/FSDP
        param_groups = [
            {"params": self.model.parameters(), "lr": self.cfg.lr},
            {"params": self.vae.parameters(), "lr": 0},
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
        )
        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer,
                **self.cfg.lr_scheduler,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def _load_tuned_state_dict(self, prefix="model."):
        ckpt = _load_checkpoint_weights_only(
            self.cfg.model.tuned_ckpt_path,
            mmap=True,
            map_location="cpu",
        )
        raw = ckpt["state_dict"]
        # torch.compile wraps self.model in OptimizedModule, saving keys as
        # "{prefix}_orig_mod.*" instead of "{prefix}*".  Detect and strip accordingly.
        # Consolidated omni exports nest the DiT one level deeper (under "model.model.")
        # alongside text_encoder/vae keys; specialists keep it under "model.".  Try the
        # deeper prefix first and pick whichever yields the DiT (patch_embedding) keys.
        for base in ("model.model.", prefix):
            for eff in (base + "_orig_mod.", base):
                cand = {k[len(eff):]: v for k, v in raw.items() if k.startswith(eff)}
                if any(k.startswith("patch_embedding") for k in cand):
                    del ckpt
                    gc.collect()
                    return cand
        # fallback: original single-prefix behavior
        compiled_prefix = prefix + "_orig_mod."
        effective_prefix = (
            compiled_prefix
            if any(k.startswith(compiled_prefix) for k in raw)
            else prefix
        )
        state_dict = {
            k[len(effective_prefix) :]: v
            for k, v in raw.items()
            if k.startswith(effective_prefix)
        }
        del ckpt
        gc.collect()
        return state_dict

    def build_scheduler(self, is_training=True):
        # Solver
        if self.sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=self.sample_shift,
                use_dynamic_shifting=False,
            )
            if not is_training:
                scheduler.set_timesteps(
                    self.sample_steps, device=self.device, shift=self.sample_shift
                )
            timesteps = scheduler.timesteps
        elif self.sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=self.sample_shift,
                use_dynamic_shifting=False,
            )
            if not is_training:
                sampling_sigmas = get_sampling_sigmas(
                    self.sample_steps, self.sample_shift
                )
                timesteps, _ = retrieve_timesteps(
                    scheduler, device=self.device, sigmas=sampling_sigmas
                )
        else:
            raise NotImplementedError("Unsupported solver.")
        return scheduler, timesteps

    def encode_text(self, texts):
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]

    def encode_video(self, videos):
        """videos: [B, C, T, H, W].

        If vae_num_views > 1, split horizontal view tiles before VAE encode
        and concatenate latent tiles back along width. This matches the original
        training path for multi-view DROID checkpoints while preserving the
        old single-view behavior when vae_num_views is absent.
        """
        if self.vae_num_views > 1:
            V = self.vae_num_views
            W = videos.shape[-1]
            assert W % V == 0, f"W={W} not divisible by vae_num_views={V}"
            W_pv = W // V
            latents = []
            for view_idx in range(V):
                chunk = videos[..., view_idx * W_pv : (view_idx + 1) * W_pv]
                latents.append(
                    self.vae.encode(
                        chunk.to(dtype=self.vae_mean.dtype),
                        [self.vae_mean, self.vae_inv_std],
                    )
                )
            return torch.cat(latents, dim=-1)
        return self.vae.encode(
            videos.to(dtype=self.vae_mean.dtype), [self.vae_mean, self.vae_inv_std]
        )

    def decode_video(self, zs):
        if self.vae_num_views > 1:
            V = self.vae_num_views
            W_lat = zs.shape[-1]
            assert W_lat % V == 0, f"W_lat={W_lat} not divisible by vae_num_views={V}"
            W_lat_pv = W_lat // V
            pixels = []
            for view_idx in range(V):
                chunk = zs[..., view_idx * W_lat_pv : (view_idx + 1) * W_lat_pv]
                pixels.append(
                    self.vae.decode(
                        chunk.to(dtype=self.vae_mean.dtype),
                        [self.vae_mean, self.vae_inv_std],
                    )
                )
            return torch.cat(pixels, dim=-1).clamp_(-1, 1)
        return self.vae.decode(
            zs.to(dtype=self.vae_mean.dtype), [self.vae_mean, self.vae_inv_std]
        ).clamp_(-1, 1)

    def flow_decode(self, latents):
        """Decode latents through flow decoder → optical flow [B, 2, T_out, H, W].

        VAE decode gives (T_lat-1)*4+1 frames; slice first frame → (T_lat-1)*4.
        """
        if self.flow_decoder_vae is None:
            raise RuntimeError(
                "Flow decoder requested but not initialized. "
                "Enable `algorithm.flow_decoder.enabled=true` and set a valid flow decoder checkpoint."
            )
        flow_out = self.flow_decoder_vae.decode(
            latents, [self.flow_vae_mean, self.flow_vae_inv_std]
        )
        return flow_out[:, :, 1:] / self.flow_train_multiplier

    @torch.no_grad()
    def decode_latents(self, video_lat, decode_outputs=None):
        if decode_outputs is None:
            outputs = self.decode_outputs_cfg
        elif isinstance(decode_outputs, str):
            outputs = [decode_outputs]
        else:
            outputs = list(decode_outputs)
        outputs = set(outputs)

        decoded = {}
        if "rgb" in outputs:
            video_pred = self.decode_video(video_lat)
            decoded["rgb"] = rearrange(video_pred, "b c t h w -> b t c h w")

        if "flow" in outputs or "flow_rgb" in outputs:
            self._ensure_flow_decoder_device()
            flow = self.flow_decode(video_lat)
            if "flow" in outputs:
                decoded["flow"] = rearrange(flow, "b c t h w -> b t c h w")
            if "flow_rgb" in outputs:
                decoded["flow_rgb"] = rearrange(
                    flow_to_rgb(flow), "b c t h w -> b t c h w"
                )

        return decoded

    @torch.no_grad()
    def prepare_embeds(self, batch):
        videos = batch["videos"]
        batch_size, t_pix, _, h, w = videos.shape

        if self.M is None and t_pix != self.n_frames:
            raise ValueError(f"Number of frames in videos must be {self.n_frames}")
        if h != self.height or w != self.width:
            raise ValueError(
                f"Height and width of videos must be {self.height} and {self.width}"
            )

        # Text embeddings
        if self.skip_text_encoder:
            prompt_embeds = [
                torch.zeros(
                    1,
                    self.cfg.text_encoder.text_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(batch_size)
            ]
        elif not self.cfg.load_prompt_embed:
            prompt_embeds = self.encode_text(batch["prompts"])
        else:
            prompt_embeds = batch["prompt_embeds"].to(self.dtype)
            prompt_embed_len = batch["prompt_embed_len"]
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, prompt_embed_len)]

        if self.M is not None:
            video_lat, K = self._sample_and_encode_window(batch, batch_size, t_pix)
            batch["K"] = K
        else:
            if "video_latents" in batch and batch["video_latents"] is not None:
                video_lat = batch["video_latents"].to(self.device)
            else:
                video_lat = self.encode_video(
                    rearrange(videos, "b t c h w -> b c t h w")
                )
            batch["K"] = 0

        batch["prompt_embeds"] = prompt_embeds
        batch["video_lat"] = video_lat
        batch["image_embeds"] = None
        batch["clip_embeds"] = None

        return batch

    def _sample_and_encode_window(self, batch, batch_size, t_pix):
        """Sample a K+M latent window per batch and encode only that sub-clip.

        For the WAN CausalVAE with temporal stride s, latent frame k maps to:
          - k == 0: pixel frame 0
          - k > 0: pixel frames [(k-1)*s + 1, k*s]

        So for a window of K+M latents starting at lat_start, the corresponding
        pixel sub-clip starts at pix_start = 0 if lat_start == 0 else (lat_start-1)*s+1
        and spans n_pix = 1 + (K+M-1)*s frames.  Encoding this sub-clip is equivalent
        to encoding the full video and slicing latents, as long as training and inference
        use the same convention (each context window is encoded independently).

        Returns:
            video_lat: [B, C, K+M, H_lat, W_lat]
            K: int
        """
        M, N = self.M, self.N
        stride = self.vae_stride[0]
        videos = batch["videos"]
        has_precomputed = (
            "video_latents" in batch and batch["video_latents"] is not None
        )

        if "src_n_frames" in batch:
            src_n = batch["src_n_frames"].cpu().numpy().astype(int)
            if has_precomputed:
                content_pix = src_n
                max_lat_total = batch["video_latents"].shape[2]
            else:
                content_pix = np.minimum(src_n, t_pix)
                max_lat_total = 1 + (t_pix - 1) // stride
            content_lens_lat = np.minimum(
                1 + (content_pix - 1) // stride, max_lat_total
            )
        else:
            total_lat = (
                batch["video_latents"].shape[2]
                if has_precomputed
                else 1 + (t_pix - 1) // stride
            )
            content_lens_lat = np.full(batch_size, total_lat, dtype=int)

        # 50% chance K = max_K (full context), 50% chance K ~ Uniform(1, max_K-1).
        min_content = int(content_lens_lat.min())
        max_K = min(N, min_content - M)
        if max_K <= 1:
            K = 1
        elif np.random.rand() < 0.5:
            K = max_K
        else:
            K = np.random.randint(1, max_K)

        lat_starts = [
            int(np.random.randint(0, max(int(content_lens_lat[bi]) - K - M, 0) + 1))
            for bi in range(batch_size)
        ]

        if has_precomputed:
            video_lat_full = batch["video_latents"].to(self.device)
            sub_lats = [
                video_lat_full[bi : bi + 1, :, lat_starts[bi] : lat_starts[bi] + K + M]
                for bi in range(batch_size)
            ]
            video_lat = torch.cat(sub_lats, dim=0)
        else:
            n_pix_window = 1 + (K + M - 1) * stride
            sub_videos = []
            for bi in range(batch_size):
                lat_start = lat_starts[bi]
                pix_start = 0 if lat_start == 0 else (lat_start - 1) * stride + 1
                sub_videos.append(
                    videos[bi : bi + 1, pix_start : pix_start + n_pix_window]
                )
            sub_videos = torch.cat(sub_videos, dim=0)
            video_lat = self.encode_video(
                rearrange(sub_videos, "b t c h w -> b c t h w")
            )

        return video_lat, K

    def add_training_noise(self, video_lat, K):
        """Apply continuous diffusion-forcing noise to a pre-sliced K+M latent window.

        Window sampling (K, lat_start) is done earlier in _sample_and_encode_window,
        so video_lat is already [B, C, K+M, H, W].

        Context frames ([:K]) share one noise level per sample, independent of the
        future. Future frames ([K:]) share a separate noise level per sample.
        `clean_hist_prob` may zero-out the first context frame's noise level.

        Returns:
            noisy_lat: [B, C, K+M, H, W]
            noise: [B, C, K+M, H, W]
            t: [B, K+M] in [0, num_train_timesteps]
        """
        b = video_lat.shape[0]
        M = self.M
        device = video_lat.device

        noise = torch.randn_like(video_lat)

        t = torch.zeros(b, K + M, device=device, dtype=torch.float32)
        context_t = torch.rand(b, device=device)
        t[:, :K] = context_t.unsqueeze(1).expand(-1, K)
        if K > 0 and np.random.rand() < self.diffusion_forcing.clean_hist_prob:
            t[:, 0] = 0.0
        future_t = torch.rand(b, device=device)
        t[:, K:] = future_t.unsqueeze(1).expand(-1, M)

        # Apply shift: t_shifted = t * shift / (1 + (shift - 1) * t)
        t_shifted = t * self.sample_shift / (1 + (self.sample_shift - 1) * t)
        t_expanded = (
            t_shifted.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        )  # [B, 1, K+M, 1, 1]

        noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
        t_out = t_shifted * self.num_train_timesteps

        return noisy_lat, noise, t_out

    def remove_noise(self, flow_pred, t, video_pred_lat):
        b, _, f = video_pred_lat.shape[:3]
        video_pred_lat = rearrange(video_pred_lat, "b c f h w -> (b f) c h w")
        flow_pred = rearrange(flow_pred, "b c f h w -> (b f) c h w")
        if t.ndim == 1:
            t = repeat(t, "b -> (b f)", f=f)
        elif t.ndim == 2:
            t = t.flatten()
        video_pred_lat = self.inference_scheduler.step(
            flow_pred,
            t,
            video_pred_lat,
            return_dict=False,
        )[0]
        video_pred_lat = rearrange(video_pred_lat, "(b f) c h w -> b c f h w", b=b)
        return video_pred_lat

    def _ensure_flow_decoder_device(self):
        """Move flow decoder to the correct GPU and dtype (lazy, since it's outside nn.Module tree)."""
        fd = self.flow_decoder_vae
        if fd is not None:
            p = next(fd.parameters())
            if p.device != self.device or p.dtype != self.dtype:
                fd.to(self.device, self.dtype)

    def _is_rank_zero(self):
        try:
            return self.global_rank == 0
        except Exception:
            return True

    def _log_rank_zero(self, name, value, **kwargs):
        """Log only on rank 0 to avoid distributed collectives in hot paths.

        This is intentionally non-sync by default so it can be called safely even
        when some ranks might skip or diverge around a metric log site.
        """
        if not self._is_rank_zero():
            return
        self.log(name, value, sync_dist=False, **kwargs)

    def training_step(self, batch, batch_idx=None):
        import time

        step_start = time.time()
        if self._is_rank_zero():
            print(f"[step {self.global_step}] training_step START", flush=True)
        batch = self.prepare_embeds(batch)
        clip_embeds = batch["clip_embeds"]
        image_embeds = batch["image_embeds"]
        prompt_embeds = batch["prompt_embeds"]
        video_lat = batch["video_lat"]
        K = batch["K"]

        noisy_lat, noise, t = self.add_training_noise(video_lat, K)
        flow = noise - video_lat

        tokens_per_frame = (
            self.lat_h * self.lat_w // (self.patch_size[1] * self.patch_size[2])
        )
        seq_len = video_lat.shape[2] * tokens_per_frame

        if image_embeds is not None:
            image_embeds = image_embeds[:, :, : video_lat.shape[2]]

        if self._is_rank_zero():
            print(
                f"[step {self.global_step}] WAN forward START (K={K}, M={self.M})",
                flush=True,
            )
        flow_pred = self.model(
            noisy_lat,
            t=t,
            context=prompt_embeds,
            clip_fea=clip_embeds,
            seq_len=seq_len,
            y=image_embeds,
        )
        if self._is_rank_zero():
            print(f"[step {self.global_step}] WAN forward DONE", flush=True)

        loss = torch.nn.functional.mse_loss(flow_pred[:, :, K:], flow[:, :, K:])

        self._loss_buffer.append(loss.detach())
        step_time = time.time() - step_start
        if self._is_rank_zero():
            print(
                f"[step {self.global_step}] loss={loss.item():.4f} time={step_time:.1f}s",
                flush=True,
            )
        if len(self._loss_buffer) >= self._loss_log_interval:
            avg_loss = torch.stack(self._loss_buffer).mean()
            self.log("train/loss", avg_loss, sync_dist=True)
            self.log("train/step_time_s", step_time, sync_dist=True)
            self._loss_buffer.clear()

        video_freq = self.cfg.logging.get("video_freq", 0)
        step0_vis = self.cfg.logging.get("step0_vis", False)
        should_step0_vis = self.global_step == 0 and step0_vis
        should_periodic_vis = (
            video_freq > 0
            and self.global_step > 0
            and self.global_step % video_freq == 0
        )
        if should_step0_vis or should_periodic_vis:
            torch.cuda.empty_cache()
            self.eval()
            with torch.no_grad():
                self._training_vis(video_lat, prompt_embeds, batch)
            self.train()
            gc.collect()
            torch.cuda.empty_cache()

        return loss

    @torch.no_grad()
    def sample_seq(
        self, context_lat, prompt_embeds, clip_embeds=None, image_embeds=None
    ):
        """Generate M future latent frames given context latent frames.

        Args:
            context_lat: [B, C, K, H, W] - clean context latent frames (K <= N)
            prompt_embeds: list of [seq_len, dim] per sample
        Returns:
            future_lat: [B, C, M, H, W] - denoised future latent frames
        """
        b, c, K, h, w = context_lat.shape
        M = self.M

        self.inference_scheduler, self.inference_timesteps = self.build_scheduler(False)

        tokens_per_frame = (
            self.lat_h * self.lat_w // (self.patch_size[1] * self.patch_size[2])
        )
        seq_len = (K + M) * tokens_per_frame

        # For I2V: create zero image_embeds with correct temporal dim if needed
        if self.cfg.model.model_type == "i2v":
            if image_embeds is None:
                image_embeds = torch.zeros(
                    b,
                    4 + self.lat_c,
                    K + M,
                    h,
                    w,
                    device=context_lat.device,
                    dtype=context_lat.dtype,
                )
            elif image_embeds.shape[2] != K + M:
                image_embeds = image_embeds[:, :, : K + M]
            if clip_embeds is None:
                clip_embeds = self.clip_features(
                    torch.zeros(
                        b,
                        1,
                        3,
                        self.height,
                        self.width,
                        device=context_lat.device,
                        dtype=context_lat.dtype,
                    )
                )

        # Start future from random noise
        future_lat = torch.randn(
            b, c, M, h, w, device=context_lat.device, dtype=context_lat.dtype
        )

        lang_guidance = self.lang_guidance if self.lang_guidance else 0
        hist_guidance = self.hist_guidance if self.hist_guidance else 0
        if lang_guidance:
            neg_prompt_embeds = self.encode_text([self.neg_prompt] * b)

        from tqdm import tqdm as _tqdm

        pbar = _tqdm(
            self.inference_timesteps,
            desc=f"WAN denoise (K={K} M={M} tok={seq_len})",
            leave=False,
            disable=(not self._is_rank_zero()),
        )
        for t in pbar:
            # Concat context (clean) + future (noisy)
            video_pred_lat = torch.cat([context_lat, future_lat], dim=2)

            # Timesteps: context = near-zero (clean), future = current t
            t_expanded = torch.full((b, K + M), t, device=self.device)
            t_expanded[:, :K] = self.inference_timesteps[-1]

            flow_pred = self.model(
                video_pred_lat,
                t=t_expanded,
                context=prompt_embeds,
                seq_len=seq_len,
                clip_fea=clip_embeds,
                y=image_embeds,
            )

            if lang_guidance:
                no_lang_flow_pred = self.model(
                    video_pred_lat,
                    t=t_expanded,
                    context=neg_prompt_embeds,
                    seq_len=seq_len,
                    clip_fea=clip_embeds,
                    y=image_embeds,
                )
            else:
                no_lang_flow_pred = torch.zeros_like(flow_pred)

            if hist_guidance:
                no_hist_video_pred_lat = torch.cat(
                    [torch.randn_like(context_lat), future_lat], dim=2
                )
                t_hist = t_expanded.clone()
                t_hist[:, :K] = self.inference_timesteps[0]
                no_hist_flow_pred = self.model(
                    no_hist_video_pred_lat,
                    t=t_hist,
                    context=prompt_embeds,
                    seq_len=seq_len,
                    clip_fea=clip_embeds,
                    y=image_embeds,
                )
            else:
                no_hist_flow_pred = torch.zeros_like(flow_pred)

            flow_pred = (
                flow_pred * (1 + lang_guidance + hist_guidance)
                - lang_guidance * no_lang_flow_pred
                - hist_guidance * no_hist_flow_pred
            )

            # Remove noise using scalar t (scheduler expects scalar timestep).
            # Context frames get incorrect denoising but are overwritten next iteration.
            updated_lat = self.remove_noise(flow_pred, t, video_pred_lat)
            future_lat = updated_lat[:, :, K:]

        return future_lat

    @torch.no_grad()
    def sample_seq_v2(
        self, context_lat, prompt_embeds, clip_embeds=None, image_embeds=None
    ):
        """Generate M future latent frames from clean context latents.

        This is the historical inference-time v2 path used by the OKTO WAN policy
        wrapper when `skip_text_encoder=True`. Unlike `sample_seq()`, it does not
        apply language/history guidance branches that require tokenizer-backed text
        encoding at inference time.
        """
        b, c, K, h, w = context_lat.shape
        M = self.M

        self.inference_scheduler, self.inference_timesteps = self.build_scheduler(False)

        tokens_per_frame = (
            self.lat_h * self.lat_w // (self.patch_size[1] * self.patch_size[2])
        )
        seq_len = (K + M) * tokens_per_frame

        if self.cfg.model.model_type == "i2v":
            if image_embeds is None:
                image_embeds = torch.zeros(
                    b,
                    4 + self.lat_c,
                    K + M,
                    h,
                    w,
                    device=context_lat.device,
                    dtype=context_lat.dtype,
                )
            elif image_embeds.shape[2] != K + M:
                image_embeds = image_embeds[:, :, : K + M]
            if clip_embeds is None:
                clip_embeds = self.clip_features(
                    torch.zeros(
                        b,
                        1,
                        3,
                        self.height,
                        self.width,
                        device=context_lat.device,
                        dtype=context_lat.dtype,
                    )
                )

        future_lat = torch.randn(
            b, c, M, h, w, device=context_lat.device, dtype=context_lat.dtype
        )

        from tqdm import tqdm as _tqdm

        pbar = _tqdm(
            self.inference_timesteps,
            desc=f"WAN denoise (K={K} M={M} tok={seq_len})",
            leave=False,
            disable=(not self._is_rank_zero()),
        )
        for t in pbar:
            video_pred_lat = torch.cat([context_lat, future_lat], dim=2)

            t_expanded = torch.full((b, K + M), t, device=self.device)
            t_expanded[:, :K] = self.inference_timesteps[-1]

            flow_pred = self.model(
                video_pred_lat,
                t=t_expanded,
                context=prompt_embeds,
                seq_len=seq_len,
                clip_fea=clip_embeds,
                y=image_embeds,
            )

            updated_lat = self.remove_noise(flow_pred, t, video_pred_lat)
            future_lat = updated_lat[:, :, K:]

        return future_lat

    def _should_run_validation_vis(self, batch_idx=None):
        """Control expensive validation visualization frequency.

        - `logging.step0_vis`: run once at step 0 when enabled.
        - `logging.val_vis_freq`: run every N global steps for later validation.
            - <= 0 means run on every validation trigger.
        - only visualize on first validation batch to avoid duplicate heavy AR runs.
        """
        if batch_idx is not None and batch_idx != 0:
            return False

        step0_vis = bool(self.cfg.logging.get("step0_vis", False))
        if self.global_step == 0:
            return step0_vis

        val_vis_freq = int(self.cfg.logging.get("val_vis_freq", 0))
        if val_vis_freq <= 0:
            return True
        return self.global_step % val_vis_freq == 0

    def validation_step(self, batch, batch_idx=None):
        self._validation_step(batch, batch_idx)

    @torch.no_grad()
    def _validation_step(self, batch, batch_idx=None):
        """Validation: loss on a training-style denoising pass, then optional AR vis.

        Always runs:
          - Training-style denoising step on validation data → logs validation/loss
        Gated by _should_run_validation_vis:
          - Training-style visualization (one K+M denoising pass, like _training_vis)
          - Autoregressive pixel-space visualization
        """
        import time
        from tqdm import tqdm

        # Slice batch to max_vis samples BEFORE prepare_embeds to avoid
        # encoding unused samples
        max_vis = 8
        for k in batch:
            if isinstance(batch[k], torch.Tensor) and batch[k].shape[0] > max_vis:
                batch[k] = batch[k][:max_vis]
            elif isinstance(batch[k], list) and len(batch[k]) > max_vis:
                batch[k] = batch[k][:max_vis]
        batch = self.prepare_embeds(batch)
        videos = batch["videos"]  # [B', T, C, H, W]
        prompt_embeds = batch["prompt_embeds"]
        clip_embeds = batch["clip_embeds"]
        image_embeds = batch["image_embeds"]
        video_lat = batch["video_lat"]

        tokens_per_frame = (
            self.lat_h * self.lat_w // (self.patch_size[1] * self.patch_size[2])
        )
        K_loss = batch["K"]
        noisy_lat, noise, t_loss = self.add_training_noise(video_lat, K_loss)
        flow_loss = noise - video_lat
        image_embeds_loss = (
            image_embeds[:, :, : video_lat.shape[2]]
            if image_embeds is not None
            else None
        )
        flow_pred_loss = self.model(
            noisy_lat,
            t=t_loss,
            context=prompt_embeds,
            clip_fea=clip_embeds,
            seq_len=video_lat.shape[2] * tokens_per_frame,
            y=image_embeds_loss,
        )
        self.log(
            "validation/loss",
            torch.nn.functional.mse_loss(
                flow_pred_loss[:, :, K_loss:], flow_loss[:, :, K_loss:]
            ),
            sync_dist=True,
        )

        # Training-style visualization (one denoising pass, no AR)
        self._training_vis(
            video_lat, prompt_embeds, batch, prefix_root="validation_vis"
        )

        if not self._should_run_validation_vis(batch_idx):
            return

        new_pixel_per_step = self.vae_stride[0]  # 4

        # Always provide full context (N latent frames) for best quality
        full_ctx_pixel = 1 + (self.N - 1) * self.vae_stride[0]  # e.g., 13 for N=4

        # Take context from the start of the trajectory
        start = 0
        pixel_list = videos[
            :, start : start + full_ctx_pixel
        ].clone()  # [B, full_ctx_pixel, C, H, W]

        ar_steps = self.cfg.diffusion_forcing.get("validation_ar_steps", 10)
        pbar = tqdm(
            total=ar_steps, desc="validation AR", disable=(not self._is_rank_zero())
        )

        total_time = 0
        for step_i in range(ar_steps):
            # Always use full context (N latent frames)
            ctx_pixel = full_ctx_pixel
            ar_N = self.N
            ctx_pixels = pixel_list[:, -ctx_pixel:]

            # Encode to ar_N latent frames
            ctx_lat = self.encode_video(
                rearrange(ctx_pixels, "b t c h w -> b c t h w")
            )  # [B, C, ar_N, H, W]

            # Recompute I2V conditioning from current context's first frame
            step_clip_embeds = (
                self.clip_features(ctx_pixels[:, :1])
                if hasattr(self, "clip_features")
                else None
            )

            iter_start = time.time()
            future_lat = self.sample_seq(
                ctx_lat,
                prompt_embeds,
                clip_embeds=step_clip_embeds,
                image_embeds=None,
            )  # [B, C, M, H, W]
            iter_time = time.time() - iter_start
            total_time += iter_time

            # Decode [ar_N ctx + 1st generated] latent → pixel frames
            decode_lat = torch.cat([ctx_lat, future_lat[:, :, :1]], dim=2)
            decoded_pixels = self.decode_latents(decode_lat, decode_outputs=["rgb"])[
                "rgb"
            ]

            # Take last stride (4) pixel frames as newly generated
            new_frames = decoded_pixels[:, -new_pixel_per_step:]  # [B, 4, C, H, W]
            pixel_list = torch.cat([pixel_list, new_frames], dim=1)
            del decode_lat, decoded_pixels, future_lat, ctx_lat, new_frames
            torch.cuda.empty_cache()

            pbar.update(1)

        pbar.close()

        # pixel_list: [B, 1 + ar_steps * stride, C, H, W]
        total_gen_pixel = pixel_list.shape[1]

        # GT: corresponding pixel frames from original video
        gt_end = min(start + total_gen_pixel, videos.shape[1])
        video_gt = videos[:, start:gt_end]

        # Log timing
        self.log("validation/total_time_s", total_time, sync_dist=False)
        self.log("validation/iterations", float(ar_steps), sync_dist=False)
        self.log("validation/time_per_iter_s", total_time / ar_steps, sync_dist=False)

        base_ar_caption = f"ar: ctx={full_ctx_pixel}px/{self.N}lat, {ar_steps}steps, total={total_gen_pixel}px"
        prompts = batch.get("prompts", None)
        ar_per_sample = None
        if prompts:
            ar_per_sample = [f"{base_ar_caption} | {p}" for p in prompts]
        caption = ar_per_sample[0] if ar_per_sample else base_ar_caption

        want_flow_rgb = (
            "flow_rgb" in self.decode_outputs_cfg and self.flow_decoder_vae is not None
        )
        if want_flow_rgb:
            pred_lat = self.encode_video(
                rearrange(pixel_list, "b t c h w -> b c t h w")
            )
            flow = self.decode_latents(pred_lat, decode_outputs=["flow"]).get(
                "flow", None
            )
            if flow is not None:
                total_t = pixel_list.shape[1]
                pred_flow = rearrange(flow, "b t c h w -> b c t h w")
                flow_panel = build_pred_flow_panel(
                    pred_flow, total_t, full_ctx_pixel, self.height, self.width
                )
                of = batch.get("optical_flow")
                gt_flow_panel = (
                    build_gt_flow_panel(
                        of, total_t, self.height, self.width, start=start
                    )
                    if of is not None
                    else None
                )
                panels = [(pixel_list, True), (flow_panel, False), (video_gt, True)]
                if gt_flow_panel is not None:
                    panels.append((gt_flow_panel, False))
                flow_suffix = " | pred | flow_rgb | gt | gt_flow"
                ar_flow_captions = (
                    [c + flow_suffix for c in ar_per_sample] if ar_per_sample else None
                )
                self._log_panels(
                    panels,
                    prefix=f"validation_vis/ar_{ar_steps}steps",
                    caption=caption + flow_suffix,
                    ctx_pixel_frames=full_ctx_pixel,
                    per_sample_captions=ar_flow_captions,
                )
                return

        # regular rgb logging
        self._log_panels(
            [(pixel_list, True), (video_gt, True)],
            prefix=f"validation_vis/ar_{ar_steps}steps",
            caption=caption,
            ctx_pixel_frames=full_ctx_pixel,
            per_sample_captions=ar_per_sample,
        )

    @torch.no_grad()
    def _training_vis(self, video_lat, prompt_embeds, batch, prefix_root="train_vis"):
        """v2 training visualization: pixel-space context + generate M frames.

        1. Sample context pixel frames from batch (random position)
        2. Encode to N latent frames (always full context)
        3. Generate M (=4) future latent frames
        4. Decode full [vis_N+M] latent → pixel frames (pred)
        5. GT = corresponding pixel frames from same position
        """
        import time

        M = self.M
        max_vis = 8  # limit samples for faster vis
        clip_embeds = batch.get("clip_embeds", None)
        image_embeds = batch.get("image_embeds", None)
        videos = batch["videos"][:max_vis]  # [B', T, C, H, W]
        prompt_embeds = prompt_embeds[:max_vis]
        if clip_embeds is not None:
            clip_embeds = clip_embeds[:max_vis]
        if image_embeds is not None:
            image_embeds = image_embeds[:max_vis]
        T_pixel = videos.shape[1]

        # Always use full context (N latent frames) for clearest visualization.
        vis_N = self.N
        ctx_pixel = 1 + (vis_N - 1) * self.vae_stride[0]
        total_pixel = 1 + (vis_N + M - 1) * self.vae_stride[0]

        # Random start position (uniform over whole sequence)
        max_start = T_pixel - total_pixel
        if dist.is_initialized() and dist.get_world_size() > 1:
            if self._is_rank_zero():
                start_value = np.random.randint(0, max(max_start, 0) + 1)
            else:
                start_value = 0
            start_tensor = torch.tensor(
                [start_value], device=video_lat.device, dtype=torch.long
            )
            dist.broadcast(start_tensor, src=0)
            start = int(start_tensor.item())
        else:
            start = np.random.randint(0, max(max_start, 0) + 1)

        # Context: consecutive pixel frames determined by vis_N
        ctx_pixels = videos[:, start : start + ctx_pixel]  # [B, 5, C, H, W]
        ctx_lat = self.encode_video(
            rearrange(ctx_pixels, "b t c h w -> b c t h w")
        )  # [B, C, N, H, W]

        vis_start = time.time()
        pred_future_lat = self.sample_seq(
            ctx_lat,
            prompt_embeds,
            clip_embeds=clip_embeds,
            image_embeds=image_embeds,
        )  # [B, C, M, H, W]
        vis_time = time.time() - vis_start
        self._log_rank_zero("train/vis_time_s", vis_time)

        # Decode full predicted sequence [N+M] latent → total_pixel pixel frames
        pred_full_lat = torch.cat([ctx_lat, pred_future_lat], dim=2)
        want_flow_rgb = (
            "flow_rgb" in self.decode_outputs_cfg and self.flow_decoder_vae is not None
        )
        decode_outputs = ["rgb", "flow"] if want_flow_rgb else ["rgb"]
        decoded = self.decode_latents(pred_full_lat, decode_outputs=decode_outputs)
        video_pred = decoded["rgb"]

        # GT: corresponding pixel frames from original video
        gt_end = min(start + total_pixel, T_pixel)
        video_gt = videos[:, start:gt_end]  # [B, <=total_pixel, C, H, W]

        prompts = batch.get("prompts", None)
        base_caption = (
            f"ctx={ctx_pixel}px/{vis_N}lat, pred={M}lat/{total_pixel - ctx_pixel}px"
        )
        per_sample_captions = None
        if prompts:
            n = min(len(prompts), max_vis)
            per_sample_captions = [f"{base_caption} | {prompts[i]}" for i in range(n)]
        caption = per_sample_captions[0] if per_sample_captions else base_caption

        if want_flow_rgb and "flow" in decoded:
            pred_flow = rearrange(decoded["flow"], "b t c h w -> b c t h w")
            total_T = video_pred.shape[1]
            flow_panel = build_pred_flow_panel(
                pred_flow, total_T, ctx_pixel, self.height, self.width
            )
            of = batch.get("optical_flow")
            gt_flow_panel = (
                build_gt_flow_panel(
                    of[:max_vis], total_T, self.height, self.width, start=start
                )
                if of is not None
                else None
            )
            panels = [(video_pred, True), (flow_panel, False), (video_gt, True)]
            if gt_flow_panel is not None:
                panels.append((gt_flow_panel, False))
            flow_suffix = " | pred | flow_rgb | gt | gt_flow"
            flow_captions = (
                [c + flow_suffix for c in per_sample_captions]
                if per_sample_captions
                else None
            )
            self._log_panels(
                panels,
                prefix=f"{prefix_root}/ctx{vis_N}_pred{M}",
                caption=caption + flow_suffix,
                ctx_pixel_frames=ctx_pixel,
                per_sample_captions=flow_captions,
            )
        else:
            self._log_panels(
                [(video_pred, True), (video_gt, True)],
                prefix=f"{prefix_root}/ctx{vis_N}_pred{M}",
                caption=caption,
                ctx_pixel_frames=ctx_pixel,
                per_sample_captions=per_sample_captions,
            )

    def _log_video_batch(self, videos_dict, fps=10, caption=None, captions=None):
        """Log multiple videos to wandb in a single call.

        Uses commit=False so video data is attached to the current wandb step
        without advancing the step counter; Lightning commits on its next flush.

        Args:
            videos_dict: {key: tensor} where tensor is [T, C, H, W] or [B, T, C, H, W]
            fps: frame rate
            caption: single caption for all videos (ignored if captions is set)
            captions: {key: caption} per-video captions
        """
        import wandb as wb

        if (
            self.logger is None
            or not hasattr(self.logger, "experiment")
            or not hasattr(self.logger.experiment, "log")
        ):
            return

        log_dict = {}
        for key, video in videos_dict.items():
            if isinstance(video, torch.Tensor):
                video = video.detach().cpu().float().numpy()
            if video.dtype != np.uint8:
                video = np.clip(video, a_min=0, a_max=1) * 255
                video = video.astype(np.uint8)
            cap = captions.get(key, caption) if captions else caption
            log_dict[key] = wb.Video(video, fps=fps, format="mp4", caption=cap)

        self.logger.experiment.log(log_dict, commit=False)
        logging.info(
            f"Logged {len(log_dict)} videos to wandb at step {self.global_step}"
        )

    def _gather_videos(self, video):
        """Gather videos across GPUs. Returns [P*B, T, C, H, W]."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            return rearrange(self.all_gather(video), "p b ... -> (p b) ...")
        return video  # single GPU: already [B, T, C, H, W]

    def _log_panels(
        self, panels, prefix, caption, ctx_pixel_frames=0, per_sample_captions=None
    ):
        """Concatenate video panels side by side and log to wandb.

        Args:
            panels: list of (tensor [B, T, C, H, W], needs_scale: bool).
                    Tensors in [-1, 1] when needs_scale=True, [0, 1] otherwise.
            prefix: wandb key prefix.
            caption: video caption (fallback when per_sample_captions is None).
            ctx_pixel_frames: leading frames of the first panel to mark with a red border.
            per_sample_captions: optional list of str, one caption per sample in the batch.
        """
        processed = [v.cpu() * 0.5 + 0.5 if s else v.cpu() for v, s in panels]
        max_t = max(v.shape[1] for v in processed)
        processed = [pad_temporal(v, max_t) for v in processed]

        if ctx_pixel_frames > 0:
            processed[0] = processed[0].clone()
            for bi in range(processed[0].shape[0]):
                add_red_border(processed[0][bi, :ctx_pixel_frames])

        video_vis = torch.cat(processed, dim=-1)
        video_vis = self._gather_videos(video_vis)

        if (
            per_sample_captions is not None
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, per_sample_captions)
            per_sample_captions = [c for sublist in gathered for c in sublist]

        if self._is_rank_zero():
            n_vis = min(len(video_vis), 8)
            captions_dict = None
            if per_sample_captions is not None:
                captions_dict = {
                    f"{prefix}/traj_{i}": per_sample_captions[i]
                    for i in range(min(n_vis, len(per_sample_captions)))
                }
            self._log_video_batch(
                {f"{prefix}/traj_{i}": video_vis[i] for i in range(n_vis)},
                fps=self.cfg.logging.fps,
                caption=caption,
                captions=captions_dict,
            )

        # Barrier so rank 0's wandb upload doesn't stall other ranks waiting
        # for the next collective (e.g. sync_dist allreduce for validation/loss).
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

    def visualize(self, video_pred, batch, prefix="validation_vis"):
        video_gt = batch["videos"]

        if self.cfg.logging.video_type == "single":
            video_vis = video_pred.cpu() * 0.5 + 0.5
            video_vis = self._gather_videos(video_vis)
            if dist.is_initialized():
                all_prompts = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(all_prompts, batch["prompts"])
                all_prompts = [item for sublist in all_prompts for item in sublist]
            else:
                all_prompts = batch["prompts"]
            if self._is_rank_zero():
                self._log_video_batch(
                    {
                        f"{prefix}/video_pred_{i}": video_vis[i]
                        for i in range(min(len(video_vis), 8))
                    },
                    fps=self.cfg.logging.fps,
                    captions={
                        f"{prefix}/video_pred_{i}": all_prompts[i]
                        for i in range(min(len(video_vis), 8))
                    },
                )
        else:
            self._log_panels(
                [(video_pred, True), (video_gt, True)],
                prefix=prefix,
                caption=None,
            )

    def maybe_reset_socket(self):
        if not self.socket:
            ctx = zmq.Context()
            socket = ctx.socket(zmq.ROUTER)
            socket.setsockopt(zmq.ROUTER_HANDOVER, 1)
            socket.bind(f"tcp://*:{self.cfg.serving.port}")
            self.socket = socket

            print(f"Server ready on port {self.cfg.serving.port}...")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """
        This function is used to test the model.
        It will receive an image and a prompt from remote gradio and generate a video.
        The remote client shall run scripts/inference_client.py to send requests to this server.
        """

        # Only rank zero sets up the socket
        if self._is_rank_zero():
            self.maybe_reset_socket()

        print(f"Waiting for request on local rank: {dist.get_rank()}")
        if self._is_rank_zero():
            ident, payload = self.socket.recv_multipart()
            request = msgpack.unpackb(payload, raw=False)
            print(f"Received request with prompt: {request['prompt']}")

            # Prepare data to broadcast
            image_bytes = request["image"]
            prompt = request["prompt"]
            data_to_broadcast = [image_bytes, prompt]
        else:
            data_to_broadcast = [None, None]

        # Broadcast the image and prompt to all ranks
        dist.broadcast_object_list(data_to_broadcast, src=0)
        image_bytes, prompt = data_to_broadcast
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomResizedCrop(
                    size=(self.height, self.width),
                    scale=(1.0, 1.0),
                    ratio=(self.width / self.height, self.width / self.height),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            ]
        )
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(pil_image)
        batch["videos"][:, 0] = image[None]

        prompt_segments = prompt.split("<sep>")
        return_flow_video = (
            "flow" in self.decode_outputs_cfg or "flow_rgb" in self.decode_outputs_cfg
        )
        hist_len = 1
        videos = batch["videos"][:, :hist_len]
        flow_videos = None
        if return_flow_video:
            b, t, _, h, w = videos.shape
            flow_videos = torch.full(
                (b, t, 3, h, w), -1.0, device=videos.device, dtype=videos.dtype
            )
        for i, prompt in enumerate(prompt_segments):
            # extending the video until all prompt segments are used
            print(f"Generating task {i+1} out of {len(prompt_segments)} sub-tasks")
            batch["prompts"] = [prompt] * batch["videos"].shape[0]
            batch["videos"][:, :hist_len] = videos[:, -hist_len:]
            seg_lat = self.sample_seq(batch, hist_len)
            decoded = self.decode_latents(
                seg_lat,
                decode_outputs=["rgb", "flow_rgb"] if return_flow_video else ["rgb"],
            )
            seg_rgb = decoded["rgb"]
            videos = torch.cat([videos, seg_rgb], dim=1)
            if return_flow_video:
                seg_flow_rgb = decoded["flow_rgb"]  # [B, T_flow, 3, H, W], [0, 1]
                seg_flow_video = torch.full_like(seg_rgb, -1.0)
                flow_steps = min(seg_flow_rgb.shape[1], max(seg_rgb.shape[1] - 1, 0))
                if flow_steps > 0:
                    seg_flow_video[:, 1 : 1 + flow_steps] = (
                        seg_flow_rgb[:, :flow_steps] * 2 - 1
                    )
                flow_videos = torch.cat([flow_videos, seg_flow_video], dim=1)
            videos = torch.clamp(videos, -1, 1)
            if return_flow_video:
                flow_videos = torch.clamp(flow_videos, -1, 1)
            hist_len = self.sliding_hist
        videos = rearrange(self.all_gather(videos), "p b t c h w -> (p b) t h w c")
        videos = videos.float().cpu().numpy()
        if return_flow_video:
            flow_videos = rearrange(
                self.all_gather(flow_videos), "p b t c h w -> (p b) t h w c"
            )
            flow_videos = flow_videos.float().cpu().numpy()

        # Only rank zero sends the reply
        if self._is_rank_zero():
            videos = np.clip(videos * 0.5 + 0.5, 0, 1)
            videos = (videos * 255).astype(np.uint8)
            # Convert videos to mp4 bytes using the utility function
            video_bytes_list = [
                numpy_to_mp4_bytes(video, fps=self.cfg.logging.fps) for video in videos
            ]
            reply = {"videos": video_bytes_list}
            if return_flow_video:
                flow_videos = np.clip(flow_videos * 0.5 + 0.5, 0, 1)
                flow_videos = (flow_videos * 255).astype(np.uint8)
                flow_video_bytes_list = [
                    numpy_to_mp4_bytes(video, fps=self.cfg.logging.fps)
                    for video in flow_videos
                ]
                reply["flow_videos"] = flow_video_bytes_list

            # Send the reply
            self.socket.send_multipart([ident, msgpack.packb(reply)])
            print(f"Sent reply to {ident}")

            self.log_video(
                "test_vis/video_pred",
                rearrange(videos, "b t h w c -> b t c h w"),
                fps=self.cfg.logging.fps,
                caption="<sep>\n".join(prompt_segments),
            )
