# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Utilities for human visualization of images and videos."""

from copy import deepcopy
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

matplotlib.use("Agg", force=True)

# TODO: Move to core


def _flow_hsv_rgb(
    flow: npt.NDArray,
    *,
    max_magnitude: float | None = None,
    percentile: float | None = 99.0,
    eps: float = 1e-12,
) -> npt.NDArray[np.uint8]:
    """
    Convert optical flow to RGB using HSV color encoding.

    Hue=angle, Value=|flow| scale, Saturation=1.

    Args:
    ----
        flow: Optical flow array of shape (H, W, 2).
        max_magnitude: Maximum flow magnitude for scaling. If None, uses percentile.
        percentile: Percentile for automatic scaling (0-100). If None with max_magnitude=None, uses max.
        eps: Small epsilon value to avoid division by zero.

    Returns:
    -------
        RGB image array of shape (H, W, 3) with dtype uint8.
    """
    u, v = flow[..., 0], flow[..., 1]
    mag = np.hypot(u, v)
    ang = np.arctan2(v, u)  # (-pi, pi]
    hue = (ang % (2.0 * np.pi)) / (2.0 * np.pi)  # [0,1)

    if max_magnitude is None:
        if percentile is None:
            scale = float(max(np.max(mag), eps))
        else:
            scale = float(max(np.percentile(mag, percentile), eps))
    else:
        scale = float(max(max_magnitude, eps))

    val = np.clip(mag / scale, 0.0, 1.0)
    sat = np.ones_like(val)

    # HSV -> RGB (vectorized)
    h = hue * 6.0
    i = np.floor(h).astype(np.int32)
    f = h - i

    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))

    r = np.empty_like(val)
    g = np.empty_like(val)
    b = np.empty_like(val)

    i_mod = i % 6
    m0 = i_mod == 0
    r[m0], g[m0], b[m0] = val[m0], t[m0], p[m0]
    m1 = i_mod == 1
    r[m1], g[m1], b[m1] = q[m1], val[m1], p[m1]
    m2 = i_mod == 2
    r[m2], g[m2], b[m2] = p[m2], val[m2], t[m2]
    m3 = i_mod == 3
    r[m3], g[m3], b[m3] = p[m3], q[m3], val[m3]
    m4 = i_mod == 4
    r[m4], g[m4], b[m4] = t[m4], p[m4], val[m4]
    m5 = i_mod == 5
    r[m5], g[m5], b[m5] = val[m5], p[m5], q[m5]

    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def flow_to_rgb_with_quiver(
    flow: npt.NDArray,
    *,
    hsv_max_magnitude: float | None = None,
    hsv_percentile: float | None = 99.0,
    eps: float = 1e-12,
    stride: int = 12,
    scale: float = 2.0,
    normalize: bool = False,
    arrow_color: str = "w",
    key: float | None = None,
    dpi: int = 100,
    min_quiver_magnitude: float = 1e-2,
) -> npt.NDArray[np.uint8]:
    """
    Render optical flow as HSV background with quiver arrow overlay.

    Arrows with magnitude < min_quiver_magnitude are not drawn. Arrows are tail-anchored.

    Args:
    ----
        flow: Optical flow array of shape (H, W, 2).
        hsv_max_magnitude: Maximum magnitude for HSV background scaling.
        hsv_percentile: Percentile for HSV background scaling (0-100).
        eps: Small epsilon value to avoid division by zero.
        stride: Spacing between arrows in pixels.
        scale: Arrow length scale factor.
        normalize: If True, normalize arrow lengths to unit vectors.
        arrow_color: Color of quiver arrows.
        key: If provided, adds a quiver key with this magnitude value.
        dpi: DPI for rendering.
        min_quiver_magnitude: Minimum magnitude threshold for drawing arrows.

    Returns:
    -------
        RGB image array of shape (H, W, 3) with dtype uint8.
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2, "flow must be (H, W, 2)"
    H, W, _ = flow.shape

    bg = _flow_hsv_rgb(
        flow,
        max_magnitude=hsv_max_magnitude,
        percentile=hsv_percentile,
        eps=eps,
    )

    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    X, Y = np.meshgrid(xs, ys)
    U = flow[ys[:, None], xs, 0]
    V = flow[ys[:, None], xs, 1]
    mag = np.hypot(U, V)

    if normalize:
        denom = np.maximum(mag, eps)
        U, V = U / denom, V / denom

    # ---- filter out small vectors completely
    mask = mag >= max(min_quiver_magnitude, 0.0)
    Xf = X[mask]
    Yf = Y[mask]
    Uf = U[mask]
    Vf = V[mask]

    figsize = (W / dpi, H / dpi)
    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.imshow(bg, origin="upper")

    q_kwargs = dict(
        angles="xy",
        scale_units="xy",
        scale=1.0 / max(scale, eps),
        width=0.003,
        pivot="tail",  # arrows start at the point
    )

    Q = None
    if Xf.size > 0:
        Q = ax.quiver(Xf, Yf, Uf, Vf, color=arrow_color, **q_kwargs)

    if key is not None and Q is not None:
        ax.quiverkey(Q, X=0.92, Y=1.05, U=key, label=f"{key:g} px", labelpos="E")

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(H, W, 4)
    rgb = rgba[..., :3].copy()
    return rgb


def color_image_to_PIL_image(color_image: npt.NDArray) -> Image.Image:
    """
    Convert a color image array to a PIL Image.

    Args:
    ----
        color_image: Color image array of shape (H, W, 4) with RGBA channels.

    Returns:
    -------
        PIL Image object.
    """
    assert (
        color_image.ndim == 3 and color_image.shape[-1] == 4
    ), "color_image must be (H, W, 4)"
    return Image.fromarray(color_image)


def depth_image_to_PIL_image(depth_image: npt.NDArray) -> Image.Image:
    """
    Convert a depth image array to a PIL Image.

    Depth values are clipped to [0, 5], normalized, and converted to 8-bit grayscale.

    Args:
    ----
        depth_image: Depth image array of shape (H, W, 1).

    Returns:
    -------
        PIL Image object in grayscale mode.
    """
    assert (
        depth_image.ndim == 3 and depth_image.shape[-1] == 1
    ), "depth_image must be (H, W, 1)"
    depth_image = deepcopy(depth_image)
    depth_image = depth_image.squeeze()  # Remove channel dimension
    depth_image = np.clip(depth_image, 0, 5)  # Clip inf distances
    depth_image = depth_image / 5  # Normalize
    depth_image = (depth_image * 255).astype(np.uint8)  # Turn to 8bit
    return Image.fromarray(depth_image, "L")


def optical_flow_to_PIL_image(optical_flow: npt.NDArray) -> Image.Image:
    """
    Convert an optical flow array to a PIL Image.

    Args:
    ----
        optical_flow: Optical flow array of shape (H, W, 2).

    Returns:
    -------
        PIL Image object with HSV-encoded flow and quiver arrows.
    """
    assert (
        optical_flow.ndim == 3 and optical_flow.shape[-1] == 2
    ), "optical_flow must be (H, W, 2)"
    optical_flow_image = flow_to_rgb_with_quiver(optical_flow)
    return Image.fromarray(optical_flow_image)


def images_to_gif(images: list[Image.Image], gif_path: Path) -> None:
    """
    Create an animated GIF from a list of PIL Images.

    Args:
    ----
        images: List of PIL Image objects (must be non-empty).
        gif_path: Path where the GIF will be saved.
    """
    assert len(images) > 0, "images must be a non-empty list"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=100,  # Duration of each frame in milliseconds (100ms = 10fps)
        loop=0,  # 0 means loop forever
    )


def image_dir_to_gif(image_dir: Path) -> None:
    """
    Create an animated GIF from all PNG images in a directory.

    The GIF is saved as 'animation.gif' in the same directory.

    Args:
    ----
        image_dir: Path to directory containing PNG images.
    """
    assert image_dir.is_dir(), "image_dir must be a directory"
    gif_path = image_dir / "animation.gif"
    images = [Image.open(image_path) for image_path in sorted(image_dir.glob("*.png"))]
    images_to_gif(images, gif_path)
