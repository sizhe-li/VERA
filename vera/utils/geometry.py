from typing import Tuple

import torch
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, Int64
from torch import Tensor


def compute_focus_point(
    ray_origins: Float[Tensor, "ray 3"],
    ray_directions: Float[Tensor, "ray 3"],
) -> Float[Tensor, "3"]:
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = torch.einsum("...i, ...j -> ...ij", ray_directions, ray_directions)
    n = n - torch.eye(3, dtype=ray_origins.dtype, device=ray_origins.device)[None]

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "ray row col -> row col", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, ray_origins, "batch i j, batch j -> batch i")
    rhs = reduce(rhs, "batch i -> i", "sum")

    # Left-matrix-multiply both sides by the pseudo-inverse of lhs to find p.
    return einsum(torch.pinverse(lhs), rhs, "i j, j -> i")


def homogenize_points(points: Float[Tensor, "*batch n"]) -> Float[Tensor, "*batch n+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vecs(vectors: Float[Tensor, "*batch n"]) -> Float[Tensor, "*batch n+1"]:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def unproject(
    coordinates_xy: Float[Tensor, "camera ray 2"],
    z: Float[Tensor, "camera ray"],
    intrinsics: Float[Tensor, "camera 3 3"],
) -> Float[Tensor, "camera ray 3"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates_xy = homogenize_points(coordinates_xy)
    coordinates_xyz = einsum(
        intrinsics.inverse(), coordinates_xy, "camera i j, camera ray j -> camera ray i"
    )

    # Apply the supplied depth values.
    return coordinates_xyz * z[..., None]


def transform_world2cam(
    homogeneous_world_xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points from world coordinates to camera coordinates."""
    world2cam = torch.inverse(cam2world)
    return transform_rigid(homogeneous_world_xyz, world2cam)


def transform_cam2world(
    homogeneous_camera_xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_camera_xyz, cam2world)


def transform_rigid(
    homogeneous_xyz: Float[Tensor, "*#batch 4"],
    transformation: Float[Tensor, "*#batch 4 4"],
) -> torch.Tensor:
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_xyz, "... i j, ... j -> ... i")


def get_world_rays(
    coordinates_xy: Float[Tensor, "camera ray 2"],
    intrinsics: Float[Tensor, "camera 3 3"],
    cam2world: Float[Tensor, "camera 4 4"],
) -> Tuple[
    Float[Tensor, "camera ray 3"],  # origins
    Float[Tensor, "camera ray 3"],  # directions
]:
    # Extract ray origins.
    origins = cam2world[..., :3, 3]

    # Get camera-space ray directions.
    directions = unproject(
        coordinates_xy,
        torch.ones_like(coordinates_xy[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vecs(directions)
    directions = transform_cam2world(
        directions,
        rearrange(cam2world, "camera h w -> camera () h w"),
    )

    # Tile the ray origins to have the same shape as the ray directions.
    _, num_rays, _ = directions.shape
    origins = repeat(origins, "camera xyz -> camera ray xyz", ray=num_rays)

    return origins, directions[..., :3]


def get_pixel_coordinates(
    height: int,
    width: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Float[Tensor, "H W 2"], Int64[Tensor, "H W 2"]]:
    """Get normalized (range 0 to 1) xy coordinates and row-col indices for an image."""

    # Each entry is a pixel-wise (row, col) coordinate.
    row = torch.arange(height, device=device)
    col = torch.arange(width, device=device)
    selector = torch.stack(torch.meshgrid(row, col, indexing="ij"), dim=-1)

    # Each entry is a spatial (x, y) coordinate in the range (0, 1).
    x = (col + 0.5) / width
    y = (row + 0.5) / height
    coordinates = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)

    return coordinates, selector


def deprecated_project(
    xyz_cam_hom: torch.Tensor,
    intrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Projects homogenized 3D points xyz_cam_hom in camera coordinates
    to pixel coordinates.

    Args:
        xyz_cam_hom: 3D points of shape (..., 4)
        intrinsics: camera intrinscics of shape (..., 3, 3)

    Returns:
        xy: homogeneous pixel coordinates of shape (..., 3) (final coordinate is 1)
    """
    xyw = torch.einsum("...ij,...j->...i", intrinsics, xyz_cam_hom[..., :3])
    z = xyw[..., -1:]
    xyw = xyw / (z + 1e-9)  # z-divide
    return xyw[..., :3], z


def project(
    points: Float[Tensor, "... 3"],
    intrinsics: Float[Tensor, "... 3 3"],
    epsilon: float = 1e-8,
) -> Float[Tensor, "batch point 2"]:
    points = points / (points[..., -1:] + epsilon)
    points = einsum(intrinsics, points, "... i j, ... j -> ... i")
    depth = points[..., -1:]
    return points[..., :2], depth


def lester_project(
    xyz_cam_hom: torch.Tensor,
    intrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Projects homogenized 3D points xyz_cam_hom in camera coordinates
    to pixel coordinates.

    Args:
        xyz_cam_hom: 3D points of shape (..., 4)
        intrinsics: camera intrinscics of shape (..., 3, 3)

    Returns:
        xy: homogeneous pixel coordinates of shape (..., 3) (final coordinate is 1)
    """
    xyw = torch.einsum("...ij,...j->...i", intrinsics, xyz_cam_hom[..., :3])
    depth = xyw[..., -1:]
    return xyw[..., :3], depth


def get_world_rays_with_z(
    coordinates_xy: Float[Tensor, "camera ray 2"],
    intrinsics: Float[Tensor, "camera 3 3"],
    cam2world: Float[Tensor, "camera 4 4"],
) -> Tuple[
    Float[Tensor, "camera ray 3"],  # origins
    Float[Tensor, "camera ray 3"],  # directions
    Float[Tensor, "camera ray 1"],  # z
]:
    # Extract ray origins.
    origins = cam2world[..., :3, 3]

    # Get camera-space ray directions.
    directions = unproject(
        coordinates_xy,
        torch.ones_like(coordinates_xy[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)
    z = directions[..., -1:]

    # Transform ray directions to world coordinates.
    directions = homogenize_vecs(directions)
    directions = transform_cam2world(
        directions,
        rearrange(cam2world, "camera h w -> camera () h w"),
    )

    # Tile the ray origins to have the same shape as the ray directions.
    _, num_rays, _ = directions.shape
    origins = repeat(origins, "camera xyz -> camera ray xyz", ray=num_rays)

    return origins, directions[..., :3], z


def project_world_coords_to_camera(
    coords_3d_world: Float[Tensor, "B N 3"],
    cam2world: Float[Tensor, "B 4 4"],
    intrinsics: Float[Tensor, "B 3 3"],
) -> Float[Tensor, "B N 2"]:
    c3d_world_hom = homogenize_points(coords_3d_world)
    c3d_cam_hom = transform_world2cam(c3d_world_hom, cam2world[..., None, :, :])
    c2d_cam, depth = deprecated_project(c3d_cam_hom, intrinsics.unsqueeze(1))

    return c2d_cam[..., :2]


def project_world_coords_to_camera2(
    coords_3d_world: Float[Tensor, "B N 3"],
    cam2world: Float[Tensor, "B 4 4"],
    intrinsics: Float[Tensor, "B 3 3"],
) -> Float[Tensor, "B N 2"]:
    c3d_world_hom = homogenize_points(coords_3d_world)
    c3d_cam_hom = transform_world2cam(c3d_world_hom, cam2world[..., None, :, :])
    c2d_cam, depth = deprecated_project(c3d_cam_hom, intrinsics.unsqueeze(1))

    return c2d_cam[..., :2], depth


def project_world_coords_to_camera3(
    coords_3d_world: Float[Tensor, "B N 3"],
    cam2world: Float[Tensor, "B 4 4"],
    intrinsics: Float[Tensor, "B 3 3"],
) -> Float[Tensor, "B N 2"]:
    c3d_world_hom = homogenize_points(coords_3d_world)
    c3d_cam_hom = transform_world2cam(c3d_world_hom, cam2world[..., None, :, :])
    c2d_cam, depth = lester_project(c3d_cam_hom, intrinsics.unsqueeze(1))

    return c2d_cam[..., :2], depth


def is_occluded(
    point_3d_world: Float[Tensor, "3"],
    cam2world_b: Float[Tensor, "4 4"],
    intrinsics_b: Float[Tensor, "3 3"],
    depth_map_b: Float[Tensor, "H W"],
    image_height: int,
    image_width: int,
    threshold: float = 0.05,
) -> bool:
    """
    Checks if a world-space point is occluded in view B.
    """

    # Step 1: Project point from world to camera B
    point_3d_world_h = homogenize_points(point_3d_world[None, None])  # [1, 1, 4]
    point_cam_h = transform_world2cam(
        point_3d_world_h, cam2world_b[None, None]
    )  # [1, 1, 4]
    point_cam = point_cam_h[..., :3]  # [1, 1, 3]
    z_proj = point_cam[..., 2].item()

    # Step 2: Project to pixel space
    point_2d_hom, _ = deprecated_project(point_cam_h, intrinsics_b[None])  # [1, 1, 3]
    x, y = point_2d_hom[0, 0, 0], point_2d_hom[0, 0, 1]

    # Step 3: Check if pixel is within bounds
    px, py = int(round(x.item())), int(round(y.item()))
    if not (0 <= px < image_width and 0 <= py < image_height):
        return True  # Consider out-of-frame as occluded

    # Step 4: Compare with depth map
    depth_at_pixel = depth_map_b[py, px].item()

    # Step 5: Occlusion test
    is_occ = z_proj > depth_at_pixel + threshold
    return is_occ
