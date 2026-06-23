from __future__ import annotations

from typing import Optional, Sequence

from transformers import Dinov2Config, DPTConfig


def build_dpt_config(
    image_size: int,
    backbone_preset: str = "small",
    out_indices: Optional[Sequence[int]] = None,
    neck_preset: Optional[str] = None,
    neck_hidden_sizes: Optional[Sequence[int]] = None,
) -> DPTConfig:
    backbone_model_map = {
        "small": "facebook/dinov2-small",
        "base": "facebook/dinov2-base",
        "large": "facebook/dinov2-large",
        "giant": "facebook/dinov2-giant",
    }
    backbone_model = backbone_model_map.get(
        str(backbone_preset).lower(), backbone_model_map["small"]
    )
    if out_indices is None:
        out_indices = [1, 2, 3, 4]

    backbone_config = Dinov2Config.from_pretrained(
        backbone_model,
        out_features=["stage1", "stage2", "stage3", "stage4"],
        reshape_hidden_states=False,
        out_indices=list(out_indices),
    )

    if neck_hidden_sizes is None:
        if neck_preset is not None:
            preset_map = {
                "S": [128, 128],
                "M": [96, 96, 128, 128],
                "L": [128, 192, 192, 256],
                "XL": [256, 256, 384, 384],
            }
            neck_hidden_sizes = preset_map.get(str(neck_preset).upper())
        if neck_hidden_sizes is None:
            default_sizes = [96, 96, 128, 128]
            neck_hidden_sizes = default_sizes[: len(out_indices)]

    return DPTConfig(
        backbone_config=backbone_config,
        image_size=image_size,
        neck_hidden_sizes=list(neck_hidden_sizes),
    )
