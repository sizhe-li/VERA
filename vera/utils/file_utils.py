# okto/utils/file_utils.py
from __future__ import annotations

import gzip
import io
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import zarr
from PIL import Image


# ---------------------------------------------------------------------
# Zarr utilities
# ---------------------------------------------------------------------
def load_zarr_file(filename: Path) -> Dict[str, np.ndarray]:
    """Load all datasets from a Zarr group (or a single Zarr array) into a dictionary."""
    try:
        root = zarr.open(str(filename), mode="r")
    except zarr.errors.GroupNotFoundError as e:
        p = Path(filename)
        # If we have tracks/visibility dirs but no metadata, rsync may have excluded files
        has_v2_meta = (p / ".zgroup").exists()
        has_v3_meta = (p / "zarr.json").exists()
        if not has_v2_meta and not has_v3_meta and (p / "tracks").is_dir():
            raise zarr.errors.GroupNotFoundError(
                f"{e}\n"
                "Hint: Zarr metadata is missing (v2: .zgroup/.zarray, v3: zarr.json). "
                "If copied via rsync, check excludes: --exclude='.*' skips dotfiles, "
                "--exclude='*.json' skips zarr.json. Re-copy without those excludes."
            ) from e
        raise
    if isinstance(root, zarr.Array):
        # Single array (e.g. older zarr or single-array store)
        key = Path(filename).stem
        return {key: np.array(root)}
    # Group: multiple arrays
    return {key: np.array(root[key]) for key in root.array_keys()}


def _blosc_compressors_v3():
    """Zarr v3: list of codecs for Blosc (zstd, clevel=3, shuffle). Returns None if v3 codecs unavailable."""
    try:
        from zarr.codecs import BloscCodec

        return [BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")]
    except ImportError:
        return None


def save_zarr_file(
    filename: Path,
    data: Dict[str, np.ndarray],
    *,
    dtype: Optional[Union[np.dtype, type]] = None,
    compressor: Optional[Any] = None,
):
    """Save a dict of arrays into a Zarr group on disk.

    Args:
        filename: Path to the zarr group (directory).
        data: Dict mapping array names to numpy arrays.
        dtype: If set, cast each array to this dtype before saving (e.g. np.float16 for smaller storage).
        compressor: If set, use Blosc compression. Zarr v3 uses codecs; zarr v2 uses numcodecs.Blosc.
    """
    store = zarr.storage.LocalStore(str(filename))
    root = zarr.group(store=store, overwrite=True)
    use_compression = compressor is not None
    compressors_v3 = _blosc_compressors_v3() if use_compression else None
    for key, arr in data.items():
        arr = np.asarray(arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        if use_compression and compressors_v3 is not None:
            root.create_array(key, data=arr, compressors=compressors_v3)
        elif use_compression:
            root.create_array(key, data=arr, compressor=compressor)
        else:
            root.create_array(key, data=arr)


# ---------------------------------------------------------------------
# Gzip + pickle utilities (cross-framework safe)
# ---------------------------------------------------------------------
class CPU_Unpickler(pickle.Unpickler):
    """Safe unpickler that forces Torch tensors and JAX arrays onto CPU."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        elif module == "jax.interpreters.xla" and name == "DeviceArray":
            # Gracefully skip JAX DeviceArrays
            return lambda *args, **kwargs: None
        else:
            return super().find_class(module, name)


def load_gzip_file(filename: str | Path) -> Any:
    """Load a gzip-compressed pickle file onto CPU."""
    with gzip.open(filename, "rb") as f:
        return CPU_Unpickler(f).load()


def save_gzip_file(filename: str | Path, data: Any):
    """Save Python data using gzip + pickle protocol 4."""
    if isinstance(filename, Path):
        filename = str(filename)
    with gzip.open(filename, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load_image_file_to_torch(
    image_filename,
    rescale_size: None | Tuple[int, int] = None,
    center_crop_size: None | Tuple[int, int] = None,
):
    rgb = load_numpy_image(
        image_filename, rescale_size=rescale_size, center_crop_size=center_crop_size
    )
    rgb = numpy_to_torch_image(rgb)
    return rgb


def torch_to_numpy_image(torch_image):
    """
    Convert a torch image tensor (C, H, W) in [0,1] to a NumPy image (H, W, C) in [0,255].
    """
    # Ensure tensor is on CPU and detached from graph
    image = torch_image.detach().cpu()

    # Permute to (H, W, C)
    image = image.permute(1, 2, 0)

    # Clamp to [0,1] to avoid overflow, then scale to [0,255]
    image = (image.clamp(0, 1).numpy() * 255).astype("uint8")

    return image


def numpy_to_torch_image(numpy_image):
    image = torch.from_numpy(numpy_image.astype("float32") / 255.0)
    image = image[:, :, :3]

    image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return image


def numpy_to_torch_video(numpy_video: List[np.ndarray]):
    video_th = torch.from_numpy(np.stack(numpy_video, axis=0).astype("float32") / 255.0)
    video_th = video_th[:, :, :, :3]
    video_th = video_th.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

    return video_th


def load_numpy_image(
    image_filename,
    rescale_size: None | Tuple[int, int] = None,
    center_crop_size: None | Tuple[int, int] = None,
):
    pil_image = Image.open(image_filename)
    if rescale_size is not None:
        h_scaled, w_scaled = rescale_size
        pil_image = pil_image.resize((w_scaled, h_scaled), resample=Image.LANCZOS)
        if center_crop_size is not None:
            h_new, w_new = center_crop_size
            x_min = (w_scaled - w_new) // 2
            y_min = (h_scaled - h_new) // 2
            pil_image = pil_image.crop((x_min, y_min, x_min + w_new, y_min + h_new))

    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(3, axis=2)
    assert len(image.shape) == 3
    assert image.dtype == np.uint8
    assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
    return image
