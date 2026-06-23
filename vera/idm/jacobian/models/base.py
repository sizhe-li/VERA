from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from jaxtyping import Float
from vera.utils import convention
from vera.utils.geometry import project_world_coords_to_camera
from torch import Tensor, nn

T = TypeVar("T")


# -----------------------------------------------------------------------------
# Base model configuration
# -----------------------------------------------------------------------------
@dataclass
class BaseModelCfg:
    name: Literal[""]
    command_dim: int
    spatial_dim: int


@dataclass
class CameraInput:
    xyz: Float[Tensor, "batch rays 3"]
    ctxt_extrinsics: Float[Tensor, "batch 4 4"]
    ctxt_intrinsics: Float[Tensor, "batch 3 3"]
    trgt_extrinsics: Float[Tensor, "batch 4 4"]
    trgt_intrinsics: Float[Tensor, "batch 3 3"]


@dataclass
class InputObservation:
    rgb: (
        Float[Tensor, "batch 3 height width"]
        | Float[Tensor, "batch views 3 height width"]
    )
    view_ids: Tensor | None = None
    camera: CameraInput | None = None
    q: Tensor | None = None


@dataclass
class InputCommand:
    du: Float[Tensor, "batch c_dim"]


@dataclass
class JacobianFieldOutput:
    jacobian: (
        Float[Tensor, "batch c_dim s_dim height width"]
        | Float[Tensor, "batch rays c_dim s_dim"]
    )
    optical_flow: (
        Float[Tensor, "batch s_dim height width"] | Float[Tensor, "batch rays 2"]
    )
    scene_flow: (
        Float[Tensor, "batch 3 height width"] | Float[Tensor, "batch rays 3"] | None
    ) = None
    flow_confidence: (
        Float[Tensor, "batch 1 height width"] | None
    ) = None


class JacobianFieldInterface(nn.Module, ABC, Generic[T]):
    """
    Abstract base for image-based Jacobian field models.
    All subclasses (Unet, Transformer, etc.) must implement compute_jacobian().
    """

    cfg: T

    def __init__(self, cfg: T):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def compute_jacobian(
        self, input_obs: InputObservation
    ) -> Float[Tensor, "batch c_dim s_dim height width"]:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input_obs: InputObservation, input_cmd: InputCommand
    ) -> JacobianFieldOutput:
        raise NotImplementedError


class SceneJacobianFieldInterface(JacobianFieldInterface[T]):
    """
    Abstract base for scene-based Jacobian field models.
    All subclasses (NeRF, etc.) must implement compute_jacobian().
    """

    @abstractmethod
    def compute_jacobian(
        self,
        input_obs: InputObservation,
    ) -> Float[Tensor, "batch channel height width"]:
        raise NotImplementedError

    def forward(
        self,
        input_obs: InputObservation,
        input_cmd: InputCommand,
    ) -> JacobianFieldOutput:
        raise NotImplementedError

    def compute_optical_flow(
        self,
        xyz: Float[Tensor, "batch ray 3"],
        scene_flow: Float[Tensor, "batch ray spatial_dim"],
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        height: int,
        width: int,
    ):
        xyz_warped = xyz + scene_flow

        intrinsics_denormalized = convention.denormalize_intrinsics(
            intrinsics, height=height, width=width
        )

        uv = project_world_coords_to_camera(
            xyz,
            extrinsics,
            intrinsics_denormalized,
        )

        uv_warped = project_world_coords_to_camera(
            xyz_warped,
            extrinsics,
            intrinsics_denormalized,
        )

        optical_flow = uv_warped - uv

        return optical_flow
