from math import sqrt
from typing import Dict, List, Union

import cv2
import numpy as np
import plotly.graph_objects as go
import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Float
from torch import Tensor

JACOBIAN_COLORMAP: Dict[str, List[List[float]]] = {
    "pusher": [
        [0, 1, 0],
        [0, 0, 1],
    ],
    "planar_hand": [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
        [1, 0.5, 1],
        [0.5, 0.5, 1],
    ],
    "shadow_finger": [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    "panda": [
        [0.0, 0.5, 0.5],  # teal
        [0.0, 1.0, 0.0],  # green
        [0.8, 0.1, 0.1],  # dark red
        [0.8, 0.0, 0.8],  # purple
        [0.0, 0.8, 0.0],  # dark green
        [1.0, 0.8, 0.0],  # orange-yellow
        [1.0, 1.0, 0.0],  # yellow
        # [1.0, 0.0, 0.0],  # red
    ],
    "iiwa": [
        [0.0, 0.5, 0.5],  # teal
        [0.0, 1.0, 0.0],  # green
        [0.8, 0.1, 0.1],  # dark red
        [0.8, 0.0, 0.8],  # purple
        [0.0, 0.8, 0.0],  # dark green
        [1.0, 1.0, 0.0],  # yellow
        # [1.0, 0.0, 0.0],  # red
    ],
    "ur5": [
        [0.0, 0.5, 0.5],  # teal
        [0.0, 1.0, 0.0],  # green
        [0.8, 0.1, 0.1],  # dark red
        [0.8, 0.0, 0.8],  # purple
        [0.0, 0.8, 0.0],  # dark green
        [1.0, 1.0, 0.0],  # yellow
    ],
    "leap": [
        [0.0, 0.5, 0.5],  # teal
        [0.0, 1.0, 0.0],  # green
        [0.8, 0.1, 0.1],  # dark red
        [0.8, 0.0, 0.8],  # magenta
        [0.0, 0.8, 0.0],  # dark green
        [1.0, 0.8, 0.0],  # orange
        [1.0, 1.0, 0.0],  # yellow
        [1.0, 0.0, 0.0],  # red
        [0.0, 0.0, 1.0],  # blue
        [0.5, 0.0, 0.5],  # purple
        [0.6, 0.3, 0.0],  # brown
        [0.0, 1.0, 1.0],  # cyan
        [0.3, 0.3, 0.3],  # dark gray
        [0.7, 0.7, 0.7],  # light gray
        [0.4, 0.2, 0.6],  # indigo
        [0.9, 0.6, 0.7],  # pink
    ],
    "allegro": [
        [0.0, 0.5, 0.5],
        [0, 1, 0],
        [0.8, 0.1, 0.1],
        [0.8, 0.0, 0.8],
        [0.0, 0.8, 0],
        [1.0, 0.8, 0],
        [1, 1, 0],
        [1, 0.0, 0.0],
    ],
    "model_allegro": [
        [0.0, 0.5, 0.5],
        [0, 1, 0],
        [0.8, 0.1, 0.1],
        [0.8, 0.0, 0.8],
        [0.0, 0.8, 0],
        [1.0, 0.8, 0],
        [1, 1, 0],
        [1, 0.0, 0.0],
    ],
    "model_allegro_transformer": [
        [0.0, 0.5, 0.5],
        [0, 1, 0],
        [0.8, 0.1, 0.1],
        [0.8, 0.0, 0.8],
        [0.0, 0.8, 0],
        [1.0, 0.8, 0],
        [1, 1, 0],
        [1, 0.0, 0.0],
    ],
    "model_hsa": [
        [0.9, 0.0, 0.0],
        [0.0, 0.9, 0.0],
        [0, 0, 1.0],
        [0.8, 0.0, 1.0],
    ],
    "pneumatic_hand_only": [
        [0, 0, 1],
        [0.9, 0.2, 0.0],
        [0, 0.9, 0],
        [1.0, 0.0, 1.0],
        [0.07, 0.63, 0.49],
        [0.35, 0.56, 0.14],
    ],
    # eef-delta gripper (mimicgen / DROID SE3): tx, ty, tz, rx, ry, rz, gripper
    "eef_gripper": [
        [1.0, 0.0, 0.0],   # tx - red
        [0.0, 1.0, 0.0],   # ty - green
        [0.0, 0.4, 1.0],   # tz - blue
        [0.0, 1.0, 1.0],   # rx - cyan
        [1.0, 0.0, 1.0],   # ry - magenta
        [1.0, 1.0, 0.0],   # rz - yellow
        [1.0, 0.6, 0.0],   # gripper - orange
    ],
}


def _default_jacobian_palette(num_cmd: int) -> List[List[float]]:
    """A distinct colored palette for robots without a registered colormap, so the Jacobian vis is
    never grayscale. Evenly spaced hues around the wheel (HSV->RGB), one per command dimension."""
    import colorsys
    return [list(colorsys.hsv_to_rgb(i / max(num_cmd, 1), 0.9, 1.0)) for i in range(int(num_cmd))]


def _to_uint8_image(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)

    # If float (assume 0..1), scale to 0..255; otherwise cast to uint8.
    if np.issubdtype(x.dtype, np.floating):
        x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8)

    return np.ascontiguousarray(x)


def _to_float_flow(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(x)


def draw_flow_arrows(
    canvas,
    flow_field,
    color: tuple = (255, 255, 255),
    threshold: float = 1.0,
    sparsity: int = 20,
    flow_scale: int = 2,
):
    """
    Draw arrows on an image given a dense flow field. Accepts either float [0,1]
    or uint8 [0,255] canvas. Returns uint8 canvas.
    """
    canvas_u8 = _to_uint8_image(canvas)
    flow = _to_float_flow(flow_field)

    assert (
        flow.shape[:2] == canvas_u8.shape[:2]
    ), f"canvas shape {canvas_u8.shape}, flow shape {flow.shape}"

    H, W = flow.shape[:2]

    for y in range(0, H, sparsity):
        for x in range(0, W, sparsity):
            dx, dy = flow[y, x]
            if abs(dx) < threshold and abs(dy) < threshold:
                continue
            dx_i = int(flow_scale * dx)
            dy_i = int(flow_scale * dy)

            # draw
            canvas_u8 = cv2.arrowedLine(
                canvas_u8,
                (x, y),
                (x + dx_i, y + dy_i),
                color=color,
                thickness=1,
                tipLength=0.3,
            )

    return canvas_u8


def compute_sensitivity(
    input_jacobian: Float[Tensor, "... C_cmd C_spatial H W"],
):
    # norm over spatial dimension
    sensitivity = input_jacobian.norm(dim=-3)

    minima = reduce(sensitivity, "... C_cmd H W -> ... C_cmd () ()", "min")
    maxima = reduce(sensitivity, "... C_cmd H W -> ... C_cmd () ()", "max")

    sensitivity = (sensitivity - minima) / (maxima - minima + 1e-10)
    sensitivity = sensitivity.clip(0, 1)
    # Gamma boost: real Jacobian sensitivity is sparse/peaky (most pixels near zero with a small
    # high-magnitude region), so a linear map renders nearly black/desaturated. gamma<1 lifts the
    # low/mid range so the colored structure is actually visible (matches the old repo's vibrancy).
    sensitivity = sensitivity ** 0.45

    return sensitivity


def visualize_sensitivity(
    input_sensitivity: Float[Tensor, "... C_cmd H W"],
    color_map: Float[Tensor, "... feature rgb"],
):
    input_sensitivity = einsum(
        input_sensitivity, color_map, "... feature H W, ... feature rgb -> ... rgb H W"
    )

    minima = reduce(input_sensitivity, "... H W -> ... () ()", "min")
    maxima = reduce(input_sensitivity, "... H W -> ... () ()", "max")

    input_sensitivity = (input_sensitivity - minima) / (maxima - minima + 1e-10)
    input_sensitivity = input_sensitivity.clip(0, 1)

    return input_sensitivity


def create_video_grid(images: Float[Tensor, "B T C H W"]):
    B, T, C, H, W = images.shape
    grid_size = int(sqrt(B))

    if grid_size**2 != B:
        raise ValueError("B must be a perfect square.")

    # Reshape to (grid_size, grid_size, T, 3, H, W)
    images = images.view(grid_size, grid_size, T, C, H, W)

    # Permute to (T, 3, grid_size, H, grid_size, W)
    images = images.permute(2, 3, 0, 4, 1, 5)

    # Reshape to (T, 3, grid_size * H, grid_size * W)
    grid = images.reshape(T, C, grid_size * H, grid_size * W)

    return grid


def plot_3d_points(points, rgb_colors=None):

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=(
                        rgb_colors if rgb_colors is not None else "blue"
                    ),  # Set color to blue
                    # opacity=0.8,
                    # line=dict(width=0.5, color='black')
                ),
            )
        ]
    )

    fig.show()


def plot_3d_points2(points, rgb_colors=None):

    # Convert tensor to numpy if needed
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if rgb_colors is not None and torch.is_tensor(rgb_colors):
        rgb_colors = rgb_colors.cpu().numpy()

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=2,
                    color=rgb_colors if rgb_colors is not None else "blue",
                ),
            )
        ]
    )

    # Compute bounding box to set equal aspect
    xyz_min = np.min(points, axis=0)
    xyz_max = np.max(points, axis=0)
    max_range = (xyz_max - xyz_min).max() / 2
    center = (xyz_min + xyz_max) / 2

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[center[0] - max_range, center[0] + max_range]),
            yaxis=dict(nticks=10, range=[center[1] - max_range, center[1] + max_range]),
            zaxis=dict(nticks=10, range=[center[2] - max_range, center[2] + max_range]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(r=10, l=10, b=10, t=10),
    )

    fig.show()


def visualize_jacobian(
    jacobian: torch.Tensor,
    robot_name: str,
    optical_flow_threshold: float = 0.1,
    flow_scale: int = 1,
) -> np.ndarray:
    """
    Public API: visualize a Jacobian field as a grid (with arrows).
    - jacobian: [1, cmd_dim, spatial_dim, H, W]
    - returns: (3, H_vis, W_vis) uint8 image
    """

    assert jacobian.shape[0] == 1, "Batch size must be 1 for visualization"
    num_cmd = jacobian.shape[-4]

    jac_vis = draw_per_channel_jacobian(jacobian, robot_name).squeeze(0)
    flow_vis = draw_per_channel_optical_flow(jacobian).squeeze(0)

    jac_vis = rearrange(jac_vis, "cmd rgb h w -> cmd h w rgb")
    jac_vis = (255 * jac_vis).clip(0, 255).to(torch.uint8).cpu().numpy()
    flow_vis = flow_vis.cpu().numpy()

    # Add arrows
    for i in range(num_cmd):
        jac_vis[i] = draw_flow_arrows(
            jac_vis[i].copy(),
            flow_vis[i],
            color=(255, 255, 255),
            threshold=optical_flow_threshold,
            flow_scale=flow_scale,
            sparsity=10,
        )

    # Make grid 2×(num_cmd/2)
    if num_cmd % 2 == 1:
        jac_vis = np.concatenate([jac_vis, np.zeros_like(jac_vis[:1])], axis=0)

    return rearrange(
        jac_vis,
        "(grid_h grid_w) h w rgb -> rgb (grid_h h) (grid_w w)",
        grid_h=2,
    )


def draw_per_channel_jacobian(jacobian, robot_name):
    """Return RGB sensitivity maps per command dimension."""
    num_cmd = jacobian.shape[-4]

    colors = JACOBIAN_COLORMAP.get(robot_name)
    if colors is None or len(colors) < num_cmd:
        # Unknown robot (e.g. "eef_gripper" before it was registered, or any new embodiment) or a
        # palette too short for the command dim -> generate a distinct colored palette instead of
        # falling back to gray.
        colors = _default_jacobian_palette(num_cmd)
    cmap = torch.tensor(
        colors,
        dtype=torch.float32,
        device=jacobian.device,
    )

    vis = []
    for i in range(num_cmd):
        mask = torch.zeros_like(jacobian)
        mask[..., i, :, :, :] = jacobian[..., i, :, :, :]
        sens = compute_sensitivity(mask)
        vis.append(visualize_sensitivity(sens, cmap))

    return torch.stack(vis, dim=-4)


def draw_per_channel_optical_flow(jacobian):
    """Compute diag du -> optical_flow per command dim."""
    num_cmd = jacobian.shape[-4]
    diag_cmds = torch.eye(num_cmd, device=jacobian.device)
    flow = einsum(jacobian, diag_cmds, "b cmd s h w, n cmd -> b n s h w")
    return rearrange(flow, "b n s h w -> b n h w s")
