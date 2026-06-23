from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def read_camera_specs(base_path: str | Path) -> dict[str, dict[str, float]]:
    """Read width/height/intrinsics from a Drake camera scenario YAML."""
    camera_specs: dict[str, dict[str, float]] = {}
    in_cameras = False
    current_camera: str | None = None
    in_focal = False

    for raw_line in Path(base_path).read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()

        if raw_line == "cameras:":
            in_cameras = True
            current_camera = None
            in_focal = False
            continue
        if in_cameras and stripped and not raw_line.startswith(" "):
            in_cameras = False
            current_camera = None
            in_focal = False

        if not in_cameras:
            continue

        if raw_line.startswith("    ") and not raw_line.startswith("        "):
            current_camera = stripped[:-1] if stripped.endswith(":") else None
            in_focal = False
            if current_camera is not None:
                camera_specs.setdefault(current_camera, {})
            continue

        if current_camera is None:
            continue

        if raw_line == "        focal:":
            in_focal = True
            continue
        if in_focal and raw_line.startswith("        ") and not raw_line.startswith(
            "            "
        ):
            in_focal = False

        if ":" not in stripped:
            continue
        key, raw_value = [piece.strip() for piece in stripped.split(":", 1)]
        if not raw_value:
            continue

        if key in {"width", "height", "center_x", "center_y"}:
            camera_specs[current_camera][key] = float(raw_value)
        elif in_focal and key in {"x", "y"}:
            camera_specs[current_camera][f"focal_{key}"] = float(raw_value)

    return camera_specs


def write_scaled_camera_scenario(
    *,
    base_path: str | Path,
    output_path: str | Path,
    target_width: int,
    target_height: int,
) -> Path:
    """Write a scenario copy with camera resolution and intrinsics scaled."""
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"Camera target size must be positive, got {target_width}x{target_height}."
        )

    base_path = Path(base_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_lines: list[str] = []
    in_cameras = False
    current_camera: str | None = None
    in_focal = False
    width_scale: float | None = None
    height_scale: float | None = None

    for raw_line in base_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()

        if raw_line == "cameras:":
            in_cameras = True
            current_camera = None
            in_focal = False
            width_scale = None
            height_scale = None
            output_lines.append(raw_line)
            continue
        if in_cameras and stripped and not raw_line.startswith(" "):
            in_cameras = False
            current_camera = None
            in_focal = False
            width_scale = None
            height_scale = None

        if (
            in_cameras
            and raw_line.startswith("    ")
            and not raw_line.startswith("        ")
        ):
            current_camera = stripped[:-1] if stripped.endswith(":") else None
            in_focal = False
            width_scale = None
            height_scale = None
            output_lines.append(raw_line)
            continue

        if current_camera is not None and raw_line == "        focal:":
            in_focal = True
            output_lines.append(raw_line)
            continue
        if in_focal and raw_line.startswith("        ") and not raw_line.startswith(
            "            "
        ):
            in_focal = False

        if not (in_cameras and current_camera is not None and ":" in stripped):
            output_lines.append(raw_line)
            continue

        indent = raw_line[: len(raw_line) - len(raw_line.lstrip(" "))]
        key, raw_value = [piece.strip() for piece in stripped.split(":", 1)]
        if not raw_value:
            output_lines.append(raw_line)
            continue

        if key == "width":
            current_width = float(raw_value)
            width_scale = float(target_width) / current_width
            output_lines.append(f"{indent}{key}: {target_width}")
            continue
        if key == "height":
            current_height = float(raw_value)
            height_scale = float(target_height) / current_height
            output_lines.append(f"{indent}{key}: {target_height}")
            continue
        if key == "center_x":
            if width_scale is None:
                raise ValueError(
                    f"Missing width before center_x in {base_path} for {current_camera}."
                )
            output_lines.append(f"{indent}{key}: {_format_float(float(raw_value) * width_scale)}")
            continue
        if key == "center_y":
            if height_scale is None:
                raise ValueError(
                    f"Missing height before center_y in {base_path} for {current_camera}."
                )
            output_lines.append(
                f"{indent}{key}: {_format_float(float(raw_value) * height_scale)}"
            )
            continue
        if in_focal and key == "x":
            if width_scale is None:
                raise ValueError(
                    f"Missing width before focal.x in {base_path} for {current_camera}."
                )
            output_lines.append(f"{indent}{key}: {_format_float(float(raw_value) * width_scale)}")
            continue
        if in_focal and key == "y":
            if height_scale is None:
                raise ValueError(
                    f"Missing height before focal.y in {base_path} for {current_camera}."
                )
            output_lines.append(
                f"{indent}{key}: {_format_float(float(raw_value) * height_scale)}"
            )
            continue

        output_lines.append(raw_line)

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return output_path


def maybe_make_scaled_camera_scenario(
    *,
    base_path: str | Path,
    target_width: int,
    target_height: int,
    enabled: bool,
    output_dir: str | Path | None = None,
    scenario_label: str = "scaled",
) -> Path:
    """Return the original scenario or a cached scaled copy."""
    resolved_base_path = Path(base_path).expanduser().resolve()
    if not enabled:
        return resolved_base_path

    camera_specs = read_camera_specs(resolved_base_path)
    if not camera_specs:
        return resolved_base_path

    unique_sizes = {
        (int(spec["width"]), int(spec["height"]))
        for spec in camera_specs.values()
        if "width" in spec and "height" in spec
    }
    if unique_sizes == {(int(target_width), int(target_height))}:
        return resolved_base_path

    runtime_root = Path(
        output_dir
        if output_dir is not None
        else os.environ.get("CRM_EPHEMERAL_TMP_ROOT", tempfile.gettempdir())
    )
    scenario_output_dir = runtime_root / "allegro_resolution_scenarios"
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    cache_key = (
        f"{resolved_base_path}:{resolved_base_path.stat().st_mtime_ns}:"
        f"{target_width}:{target_height}:{scenario_label}"
    )
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
    output_path = scenario_output_dir / (
        f"{resolved_base_path.stem}_{scenario_label}_{target_width}x{target_height}_{digest}.yaml"
    )
    if output_path.exists():
        return output_path

    return write_scaled_camera_scenario(
        base_path=resolved_base_path,
        output_path=output_path,
        target_width=target_width,
        target_height=target_height,
    )
