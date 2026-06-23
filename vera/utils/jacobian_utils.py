from math import sqrt
from typing import Dict, List, Optional, Tuple, Union

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
    "eef_gripper": [
        # Pastel palette anchored on user picks #f8f0af, #c1faf7, #e0fee0.
        # Built via build_pastel_palette.py (HSL L~0.88, S~0.85, evenly-spaced hues).
        [0.9886, 0.8282, 0.8282],   # #fcd3d3 — pastel pink
        [0.9725, 0.9412, 0.6863],   # #f8f0af — anchor: pastel yellow
        [0.8784, 0.9961, 0.8784],   # #e0fee0 — anchor: pastel mint
        [0.7569, 0.9804, 0.9686],   # #c1faf7 — anchor: pastel cyan
        [0.8282, 0.9199, 0.9886],   # #d3ebfc — pastel sky
        [0.7915, 0.7157, 0.9812],   # #cab7fa — pastel lavender
        [0.9886, 0.8282, 0.9657],   # #fcd3f6 — pastel magenta
    ],
    "panda": [
        [0.0, 0.5, 0.5],  # teal
        [0.0, 1.0, 0.0],  # green
        [0.8, 0.1, 0.1],  # dark red
        [0.8, 0.0, 0.8],  # purple
        [0.0, 0.8, 0.0],  # dark green
        [1.0, 0.8, 0.0],  # orange-yellow
        [1.0, 1.0, 0.0],  # yellow
        [1.0, 0.0, 0.0],  # red (gripper)
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
    "drake_allegro": [
        # Pastel palette anchored on user picks #f8f0af, #c1faf7, #e0fee0.
        # 16 distinct hues evenly distributed on the wheel at HSL L~0.88, S~0.85
        # (built via build_pastel_palette.py). Anchors land at slots 2, 5, 8.
        [0.9886, 0.8282, 0.8282],   # #fcd3d3 — pastel pink
        [0.9812, 0.8152, 0.7157],   # #fad0b7 — pastel peach
        [0.9725, 0.9412, 0.6863],   # #f8f0af — anchor: pastel yellow
        [0.9480, 0.9812, 0.7157],   # #f2fab7 — pastel chartreuse
        [0.9084, 0.9886, 0.8282],   # #e8fcd3 — pastel pear
        [0.8784, 0.9961, 0.8784],   # #e0fee0 — anchor: pastel mint
        [0.8282, 0.9886, 0.8683],   # #d3fcdd — pastel honeydew
        [0.7157, 0.9812, 0.8816],   # #b7fae1 — pastel seafoam
        [0.7569, 0.9804, 0.9686],   # #c1faf7 — anchor: pastel cyan
        [0.7157, 0.8816, 0.9812],   # #b7e1fa — pastel sky
        [0.8282, 0.8683, 0.9886],   # #d3ddfc — pastel periwinkle
        [0.7489, 0.7157, 0.9812],   # #bfb7fa — pastel violet
        [0.9084, 0.8282, 0.9886],   # #e8d3fc — pastel orchid
        [0.9480, 0.7157, 0.9812],   # #f2b7fa — pastel pink-purple
        [0.9886, 0.8282, 0.9485],   # #fcd3f2 — pastel rose
        [0.9812, 0.7157, 0.8152],   # #fab7d0 — pastel watermelon
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
    "visualmimic_se3": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    "visualmimic_joint23": [
        # Legs / hips (1-6): warm tones
        [1.0, 0.0, 0.0],   # red
        [1.0, 0.4, 0.0],   # orange
        [1.0, 0.7, 0.0],   # amber
        [1.0, 1.0, 0.0],   # yellow
        [0.8, 0.0, 0.0],   # dark red
        [1.0, 0.6, 0.3],   # coral
        # Torso / spine (7-12): greens, teals
        [0.0, 1.0, 0.0],   # green
        [0.0, 0.8, 0.2],   # dark green
        [0.0, 1.0, 0.5],   # spring green
        [0.0, 0.9, 0.9],   # teal
        [0.2, 1.0, 0.8],   # aquamarine
        [0.5, 1.0, 0.0],   # lime
        # Arms (13-18): blues, purples
        [0.0, 0.0, 1.0],   # blue
        [0.0, 0.5, 1.0],   # azure
        [0.3, 0.0, 1.0],   # indigo
        [0.6, 0.0, 1.0],   # violet
        [0.0, 0.3, 0.8],   # navy blue
        [0.4, 0.4, 1.0],   # periwinkle
        # Hands / head (19-23): magentas, cyans, accents
        [1.0, 0.0, 1.0],   # magenta
        [1.0, 0.0, 0.5],   # rose
        [0.0, 1.0, 1.0],   # cyan
        [1.0, 0.5, 1.0],   # pink
        [0.5, 0.0, 0.5],   # purple
    ],
}


def _hsl_to_rgb(h: float, s: float, l: float) -> List[float]:
    """Convert HSL (h in [0,1], s,l in [0,1]) to RGB [0,1]."""
    if s == 0:
        return [l, l, l]
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    return [
        _hue_to_rgb_component(p, q, h + 1 / 3),
        _hue_to_rgb_component(p, q, h),
        _hue_to_rgb_component(p, q, h - 1 / 3),
    ]


def _hue_to_rgb_component(p: float, q: float, t: float) -> float:
    t = t % 1.0
    if t < 1 / 6:
        return p + (q - p) * 6 * t
    if t < 1 / 2:
        return q
    if t < 2 / 3:
        return p + (q - p) * (2 / 3 - t) * 6
    return p


def get_latent_jacobian_colors(
    dim_z: int,
    dim_u: int,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Deterministic colormap for (dim_z, dim_u). Same (dim_z, dim_u) always yields same colors.
    Returns (colors_z, colors_u) each a list of [R,G,B] in [0,1].
    """
    n = dim_z + dim_u
    max_slots = max(128, n)
    palette: List[List[float]] = []
    for k in range(max_slots):
        hue = (k / max_slots) % 1.0
        rgb = _hsl_to_rgb(hue, 1.0, 0.5)
        palette.append(rgb)
    colors_z = [palette[k] for k in range(dim_z)]
    colors_u = [palette[dim_z + k] for k in range(dim_u)]
    return colors_z, colors_u


def visualize_latent_jacobian(
    jacobian: Union[torch.Tensor, np.ndarray],
    dim_z: int,
    dim_u: int,
    cell_size: Tuple[int, int] = (32, 32),
) -> np.ndarray:
    """
    Visualize latent Jacobian J of shape (..., dim_z, dim_u, H, W) as a single RGB image (H, W, 3).

    At each spatial pixel (h, w), the slice J[:, :, h, w] has shape (dim_z, dim_u). The pixel color
    is the weighted sum of latent-z colors and latent-u colors: aggregate over u to get weights for
    each z, aggregate over z to get weights for each u, then blend the two weighted-average colors.

    Interpretation: early in training the assignment is noisy (random-looking colors). As the model
    converges and learns structured latent beliefs over pixel space, the visualization should become
    more structured—e.g. coherent color regions over robot body parts or semantic regions.

    Returns (3, H, W) uint8 RGB. If cell_size > (1,1), the image is upscaled for visibility.
    """
    if isinstance(jacobian, torch.Tensor):
        jacobian = jacobian.detach().cpu().numpy()
    jacobian = np.asarray(jacobian, dtype=np.float32)
    while jacobian.ndim > 4:
        jacobian = jacobian[0]
    assert jacobian.ndim == 4, f"Expected (..., dim_z, dim_u, H, W), got {jacobian.shape}"
    dz, du, H, W = jacobian.shape
    assert dz == dim_z and du == dim_u

    colors_z, colors_u = get_latent_jacobian_colors(dim_z, dim_u)
    colors_z = np.array(colors_z, dtype=np.float32)  # (dim_z, 3)
    colors_u = np.array(colors_u, dtype=np.float32)  # (dim_u, 3)

    # At each (h, w): weight_z[i] = sum_j |J[i,j,h,w]|, weight_u[j] = sum_i |J[i,j,h,w]|
    # J has shape (dim_z, dim_u, H, W) -> axis 0=z, 1=u, 2=H, 3=W
    J_abs = np.abs(jacobian)
    weight_z = J_abs.sum(axis=1)   # sum over j (dim_u) -> (dim_z, H, W)
    weight_u = J_abs.sum(axis=0)   # sum over i (dim_z) -> (dim_u, H, W)
    sum_wz = weight_z.sum(axis=0, keepdims=True) + 1e-10   # (1, H, W)
    sum_wu = weight_u.sum(axis=0, keepdims=True) + 1e-10   # (1, H, W)
    weight_z = weight_z / sum_wz   # (dim_z, H, W)
    weight_u = weight_u / sum_wu   # (dim_u, H, W)

    # rgb_z[h,w] = sum_i weight_z[i,h,w] * color_z[i]  -> (H, W, 3)
    image_z = np.einsum("ihw,ic->hwc", weight_z, colors_z)
    image_u = np.einsum("jhw,jc->hwc", weight_u, colors_u)
    image = (0.5 * image_z + 0.5 * image_u).clip(0, 1)

    if cell_size[0] > 1 or cell_size[1] > 1:
        h_vis, w_vis = H * cell_size[0], W * cell_size[1]
        image = (image * 255).astype(np.uint8)
        image = cv2.resize(image, (w_vis, h_vis), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
    out = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return rearrange(out, "h w c -> c h w")


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
    flow_scale: float = 2.0,
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
            # Skip non-finite flow vectors: NaN fails the `< threshold` check
            # below (NaN comparisons are False), then int(NaN) raises and -- in
            # DDP validation -- kills rank 0 only, desyncing the other ranks
            # into a 30-min NCCL allreduce timeout (job 970511, 2026-06-12).
            if not (np.isfinite(dx) and np.isfinite(dy)):
                continue
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
    flow_scale: float = 1.0,
    grid_shape: Optional[Tuple[int, int]] = None,
    target_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Public API: visualize a Jacobian field as a grid (with arrows).
    - jacobian: [1, cmd_dim, spatial_dim, H, W]
    - grid_shape: (grid_h, grid_w) tile layout. Defaults to (2, ceil(num_cmd/2)).
      Pass a near-square shape (e.g. (4, 4) for 16-dim) to keep aspect ratio
      reasonable for high-dim Jacobians.
    - target_hw: optional (H_out, W_out) to resize the composed grid into.
      Useful when the Jacobian panel must fit a fixed slot in a larger layout.
    - returns: (3, H_vis, W_vis) uint8 image
    """

    assert jacobian.shape[0] == 1, "Batch size must be 1 for visualization"
    num_cmd = jacobian.shape[-4]

    # Use the pastel compositor (sens*color on black), NOT the legacy draw_per_channel_jacobian:
    # the legacy path's visualize_sensitivity does per-RGB min-max normalization, which collapses
    # any palette whose colors have all three RGB components non-zero (every pastel, e.g. the
    # eef_gripper palette) into grayscale — the "jacobian all gray" bug. The pastel path preserves
    # hue for ANY palette. (Pure-primary palettes like "panda" happened to survive the legacy path,
    # which is why the old mimicgen viewer looked colored.)
    jac_vis = draw_per_channel_jacobian_pastel(jacobian, robot_name).squeeze(0)
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

    if grid_shape is None:
        grid_h, grid_w = 2, (num_cmd + 1) // 2
    else:
        grid_h, grid_w = grid_shape
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"grid_shape must be positive, got {grid_shape}")
        if grid_h * grid_w < num_cmd:
            raise ValueError(
                f"grid_shape {grid_shape} cannot fit {num_cmd} cmd channels"
            )

    n_pad = grid_h * grid_w - num_cmd
    if n_pad > 0:
        jac_vis = np.concatenate(
            [jac_vis, np.zeros((n_pad, *jac_vis.shape[1:]), dtype=jac_vis.dtype)],
            axis=0,
        )

    out = rearrange(
        jac_vis,
        "(grid_h grid_w) h w rgb -> rgb (grid_h h) (grid_w w)",
        grid_h=grid_h,
    )

    if target_hw is not None:
        h_out, w_out = target_hw
        hwc = np.ascontiguousarray(out.transpose(1, 2, 0))
        hwc = cv2.resize(hwc, (w_out, h_out), interpolation=cv2.INTER_AREA)
        out = np.ascontiguousarray(hwc.transpose(2, 0, 1))
    return out


def draw_per_channel_jacobian(jacobian, robot_name):
    """Return RGB sensitivity maps per command dimension."""
    num_cmd = jacobian.shape[-4]

    cmap = torch.tensor(
        JACOBIAN_COLORMAP[robot_name],
        dtype=torch.float32,
        device=jacobian.device,
    )
    if cmap.shape[0] < num_cmd:
        raise ValueError(
            f"Colormap for robot '{robot_name}' has {cmap.shape[0]} colors, "
            f"but jacobian has {num_cmd} command dimensions."
        )
    cmap = cmap[:num_cmd]

    vis = []
    for i in range(num_cmd):
        mask = torch.zeros_like(jacobian)
        mask[..., i, :, :, :] = jacobian[..., i, :, :, :]
        sens = compute_sensitivity(mask)
        vis.append(visualize_sensitivity(sens, cmap))

    return torch.stack(vis, dim=-4)


def draw_per_channel_jacobian_pastel(jacobian, robot_name):
    """Pastel-safe per-channel sensitivity panels (alpha-blend, not min-max-per-RGB).

    The legacy `draw_per_channel_jacobian` -> `visualize_sensitivity` path does
    per-RGB-channel min-max normalization which collapses any palette with all
    three RGB components non-zero (every pastel) into grayscale. This variant:
      1. computes per-channel magnitude,
      2. min-max normalizes it to [0, 1] PER (cmd, frame),
      3. composites `sens * color` on a pure-black background.
    The pastel hue is preserved end-to-end; bg pixels stay black, peak pixels
    become the pure palette color, mid-tones land on the natural gradient.

    Input shape  : jacobian [..., K_cmd, 2, H, W]   (float)
    Output shape :         [..., K_cmd, 3, H, W]   (float in [0,1])
    """
    cmap = torch.tensor(
        JACOBIAN_COLORMAP[robot_name],
        dtype=torch.float32,
        device=jacobian.device,
    )
    K = jacobian.shape[-4]
    if cmap.shape[0] < K:
        raise ValueError(
            f"Colormap for robot '{robot_name}' has {cmap.shape[0]} colors, "
            f"but jacobian has {K} command dimensions."
        )
    cmap = cmap[:K]

    mag = jacobian.norm(dim=-3)                          # [..., K, H, W]
    flat = mag.flatten(start_dim=-2)
    mins = flat.min(dim=-1).values[..., None, None]
    maxs = flat.max(dim=-1).values[..., None, None]
    sens = ((mag - mins) / (maxs - mins + 1e-10)).clamp(0.0, 1.0)

    sens = sens.unsqueeze(-3)              # [..., K, 1, H, W]
    color = cmap[..., None, None]          # [K, 3, 1, 1]
    while color.dim() < sens.dim():
        color = color.unsqueeze(0)
    return (sens * color).clamp(0.0, 1.0)


def draw_per_channel_optical_flow(jacobian):
    """Compute diag du -> optical_flow per command dim."""
    num_cmd = jacobian.shape[-4]
    diag_cmds = torch.eye(num_cmd, device=jacobian.device)
    flow = einsum(jacobian, diag_cmds, "b cmd s h w, n cmd -> b n s h w")
    return rearrange(flow, "b n s h w -> b n h w s")
