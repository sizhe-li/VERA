from typing import Optional

import torch
from einops import rearrange, repeat
from omegaconf import DictConfig
from timm.models.vision_transformer import PatchEmbed
from torch import nn

from ..base_backbone import BaseBackbone
from .dit_base import DiTBase


class DiT3DSpatial(BaseBackbone):
    """
    3D DiT backbone operating on (B, T, C, H, W).

    - x: (B, T, C, H, W) where C can already include both
      conditioning channels (e.g. RGB) and diffused channels
      (e.g. jacobian + flow). This class does not care which
      is which; it just sees total C = x_shape[0].

    - Spatial conditioning is handled upstream (e.g. in FeatureDiffusion)
      by concatenating channels before calling this backbone.
    """

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask: bool = True,
    ):
        if use_causal_mask:
            raise NotImplementedError(
                "Causal masking is not yet implemented for DiT3D backbone"
            )

        super().__init__(
            cfg,
            x_shape,
            max_tokens,
            external_cond_dim,
            use_causal_mask,
        )

        hidden_size = cfg.hidden_size
        self.patch_size = cfg.patch_size
        channels, resolution, *_ = x_shape
        assert (
            resolution % self.patch_size == 0
        ), "Resolution must be divisible by patch size."
        self.num_patches = (resolution // self.patch_size) ** 2
        out_channels = self.patch_size**2 * channels

        # Patchify spatial dimensions per frame
        self.patch_embedder = PatchEmbed(
            img_size=resolution,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=hidden_size,
            bias=True,
        )

        # Temporal DiT core
        self.dit_base = DiTBase(
            num_patches=self.num_patches,
            max_temporal_length=max_tokens,
            out_channels=out_channels,
            variant=cfg.variant,
            pos_emb_type=cfg.pos_emb_type,
            hidden_size=hidden_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            learn_sigma=False,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        )
        self.initialize_weights()

    @property
    def in_channels(self) -> int:
        # x_shape is (C, H, W)
        return self.x_shape[0]

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    @staticmethod
    def _patch_embedder_init(embedder: PatchEmbed) -> None:
        # Initialize patch_embedder like nn.Linear (instead of nn.Conv2d):
        w = embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(embedder.proj.bias)

    def initialize_weights(self) -> None:
        self._patch_embedder_init(self.patch_embedder)

        # Initialize noise level embedding and external condition embedding MLPs:
        def _mlp_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.noise_level_pos_embedding.apply(_mlp_init)
        if self.external_cond_embedding is not None:
            self.external_cond_embedding.apply(_mlp_init)

    # -------------------------------------------------------------------------
    # Embedding dimensions
    # -------------------------------------------------------------------------

    @property
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.cfg.hidden_size

    @property
    def external_cond_emb_dim(self) -> int:
        return self.cfg.hidden_size if self.external_cond_dim else 0

    # -------------------------------------------------------------------------
    # Patch <-> image
    # -------------------------------------------------------------------------

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: patchified tensor of shape (B, num_patches, patch_size**2 * C)
        Returns:
            unpatchified tensor of shape (B, H, W, C)
        """
        return rearrange(
            x,
            "b (h w) (p q c) -> b (h p) (w q) c",
            h=int(self.num_patches**0.5),
            p=self.patch_size,
            q=self.patch_size,
        )

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)   -- can include conditioning + diffused channels
            noise_levels: (B, T, noise_level_dim or scalar index)
            external_cond: optional vector conditioning (B, T, D), if used
            external_cond_mask: optional mask for vector conditioning

        Returns:
            x_out: (B, T, C, H, W)   -- same C as input
        """
        input_batch_size = x.shape[0]

        # (B, T, C, H, W) -> (B*T, C, H, W)
        x = rearrange(x, "b t c h w -> (b t) c h w")

        # Patchify images
        x = self.patch_embedder(x)  # (B*T, P, hidden)
        # -> (B, T*P, hidden)
        x = rearrange(x, "(b t) p c -> b (t p) c", b=input_batch_size)

        # Noise level embeddings
        emb = self.noise_level_pos_embedding(noise_levels)  # (B, T, hidden)

        # Optional external conditioning (vector, not spatial)
        if external_cond is not None:
            emb = emb + self.external_cond_embedding(external_cond, external_cond_mask)

        # Repeat temporal embeddings over patches
        emb = repeat(emb, "b t c -> b (t p) c", p=self.num_patches)

        # Core DiT
        x = self.dit_base(x, emb)  # (B, T*P, hidden)

        # Unpatchify back to images
        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )  # (B*T, H, W, C)

        # (B*T, H, W, C) -> (B, T, C, H, W)
        x = rearrange(x, "(b t) h w c -> b t c h w", b=input_batch_size)

        return x


# --------------------------------------------------------------------------------------
# Simple tester for DiT3D — run only when executing this file directly
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import ipdb
    from omegaconf import OmegaConf

    print("=== DiT3D tester ===")

    # ------------------------------------------------------------
    # 1. Load YAML config (your dit3d.yaml)
    # ------------------------------------------------------------
    cfg = OmegaConf.create(
        {
            "name": "dit3d",
            "variant": "full",
            "pos_emb_type": "rope_3d",
            "patch_size": 2,
            "hidden_size": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
            "use_gradient_checkpointing": False,
        }
    )

    # ------------------------------------------------------------
    # 2. Construct model
    # ------------------------------------------------------------
    # Example: 4 channels (e.g. jacobian+flow) at 32×32 resolution

    height = width = 32
    nq = 2

    dim_flow = 2
    dim_jac = 2 * nq
    dim_rgb = 3

    dim_total = dim_flow + dim_jac + dim_rgb

    x_shape = torch.Size([dim_total, height, width])

    max_tokens = 8  # max temporal length
    external_cond_dim = 0  # no vector conditioning for this test

    model = DiT3DSpatial(
        cfg=cfg,
        x_shape=x_shape,
        max_tokens=max_tokens,
        external_cond_dim=external_cond_dim,
        use_causal_mask=False,
    )

    model.eval()
    print(model)
    print("Model created")

    # ------------------------------------------------------------
    # 3. Create dummy input
    # ------------------------------------------------------------
    B = 2  # batch size
    T = 4  # temporal frames
    C, H, W = x_shape

    x = torch.randn(B, T, C, H, W)
    noise_levels = torch.randn(B, T)

    # no external condition (vector-style)
    external_cond = None
    external_cond_mask = None

    # ------------------------------------------------------------
    # 4. Run forward
    # ------------------------------------------------------------
    with torch.no_grad():
        out = model(
            x=x,
            noise_levels=noise_levels,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
        )

    print("Input  shape :", x.shape)  # (B, T, C, H, W)
    print("Output shape:", out.shape)  # (B, T, C, H, W)

    # ------------------------------------------------------------
    # 5. Drop into IPDB for inspection
    # ------------------------------------------------------------
    print("Dropping into IPDB... inspect x, out, model etc.")
    ipdb.set_trace()
