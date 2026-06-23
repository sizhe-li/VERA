from dataclasses import dataclass
from typing import Literal

from einops import einsum, rearrange

from ..backbones.unet import UNet
from .base import (
    BaseModelCfg,
    InputCommand,
    InputObservation,
    JacobianFieldInterface,
    JacobianFieldOutput,
)
from .registry import register_model


@dataclass
class UnetJacobianFieldCfg(BaseModelCfg):
    name: Literal["unet"]


@register_model("unet", cfg_cls=UnetJacobianFieldCfg)
class UnetJacobianField(JacobianFieldInterface):

    def __init__(self, model_cfg: UnetJacobianFieldCfg):
        super(UnetJacobianField, self).__init__(cfg=model_cfg)

        self.command_dim = model_cfg.command_dim
        self.spatial_dim = model_cfg.spatial_dim

        # image -> UNet -> Jacobian field
        self.jacobian_field = UNet(
            out_channels=self.command_dim * self.spatial_dim,
            in_channels=3,
            depth=3,
            start_filts=32,
        )

    def compute_jacobian(
        self,
        input_obs: InputObservation,
    ):
        jacobian = self.jacobian_field(
            input_obs.rgb
        )  # b (command_dim * spatial_dim) h w
        jacobian = rearrange(
            jacobian,
            "b (c_dim s_dim) h w -> b c_dim s_dim h w",
            c_dim=self.command_dim,
            s_dim=self.spatial_dim,
        )

        return jacobian

    def forward(
        self, input_obs: InputObservation, input_cmd: InputCommand
    ) -> JacobianFieldOutput:
        jacobian = self.compute_jacobian(input_obs)

        flow = einsum(
            jacobian, input_cmd.du, "b c_dim s_dim h w, b c_dim -> b s_dim h w"
        )

        return JacobianFieldOutput(jacobian=jacobian, optical_flow=flow)
