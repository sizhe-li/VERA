from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ..backbones.unet import UNet
from .base import (
    BaseModelCfg,
    InputCommand,
    InputObservation,
    JacobianFieldInterface,
    JacobianFieldOutput,
)
from .registry import register_model


# initialize with a small value
def init_weights(m, std: float = 1e-2):
    if type(m) == nn.Linear:
        if m.weight is not None:
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, mean=0.0, std=std)


@dataclass
class TransformerJacobianFieldCfg(BaseModelCfg):
    name: Literal["transformer"]


@register_model("transformer", cfg_cls=TransformerJacobianFieldCfg)
class TransformerJacobianField(JacobianFieldInterface):

    def __init__(self, model_cfg: TransformerJacobianFieldCfg):
        super(TransformerJacobianField, self).__init__(cfg=model_cfg)

        self.command_dim = model_cfg.command_dim
        self.spatial_dim = model_cfg.spatial_dim

        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        # freeze the DINO model
        for param in self.dino.parameters():
            param.requires_grad = False
        self.patch_size = self.dino.patch_size

        self.decoder = UNet(
            in_channels=3 + 384,
            out_channels=(self.command_dim * self.spatial_dim),
            depth=3,
        )
        self.decoder.apply(init_weights)

    def compute_jacobian(
        self, input_obs: InputObservation
    ) -> Float[Tensor, "b c_dim s_dim h w"]:
        input_img = input_obs.rgb
        image_height, image_width = input_img.shape[-2:]

        tokens = self.dino.get_intermediate_layers(input_img)[0]
        tokens = repeat(
            tokens,
            "b (h w) c -> b c (h hps) (w wps)",
            h=image_height // self.patch_size,
            w=image_width // self.patch_size,
            hps=self.patch_size,
            wps=self.patch_size,
        )

        tokens = torch.cat([input_img, tokens], dim=1)

        jacobian = self.decoder(tokens)

        jacobian = rearrange(
            jacobian,
            "b (c_dim s_dim) h w -> b c_dim s_dim h w",
            h=image_height,
            w=image_width,
            c_dim=self.command_dim,
            s_dim=self.spatial_dim,
        )

        return jacobian

    def forward(self, input_obs: InputObservation, input_cmd: InputCommand):
        image_height, image_width = input_obs.rgb.shape[-2:]
        jacobian = self.compute_jacobian(input_obs)

        dx = einsum(
            jacobian,
            input_cmd.du,
            "b c_dim s_dim h w, b c_dim -> b s_dim h w",
        )

        return JacobianFieldOutput(
            jacobian=jacobian,
            optical_flow=dx,
        )
