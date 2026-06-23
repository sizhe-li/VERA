import collections
import copy
import concurrent.futures
import itertools
import json
import multiprocessing
import os
import time
import urllib.error
import urllib.request
import warnings
import atexit
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import gymnasium as gym
import mediapy as media
import numpy as np
import torch
import tqdm
from crm.common import load_hydra_cfg
from crm.hardware_station.station import MakeHardwareStation, load_scenario
from crm.pipeline.resolvers import register_resolvers
from omegaconf import OmegaConf
from vera.env_runner.base_runner import BaseRunner, BaseRunnerCfg, rr
from vera.policy.base_policy import BasePolicy, PolicyObservation, PolicyOutput
from vera.utils.logging import cyan
from vera.utils.drake_scenario_utils import maybe_make_scaled_camera_scenario
from PIL import Image
from pydrake.all import RenderEngineGltfClientParams, Simulator
from scenarios.util.scenario_utils import get_default_package_xmls
from tasks_diffusion_policy.neural_jacobian.allegro.env import (
    get_default_q_cameras,
    make_env,
)
from tasks_diffusion_policy.neural_jacobian.common.common_env import (
    get_color_image_port_name_from_camera_name,
)
from tasks_diffusion_policy.neural_jacobian.common.display_check import (
    if_docker_then_check_display_and_x_server,
)
from tasks_diffusion_policy.neural_jacobian.common.paths import (
    NEURAL_JACOBIAN_DIR,
)

RenderObsKey = str | list[str]

_BLENDER_RERENDER_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_BLENDER_URL_COUNTER = itertools.count()
_DEFAULT_ALLEGRO_CAMERA_ORDER = ["camera_0", "camera_1", "camera_2"]
_DEFAULT_ALLEGRO_RENDER_SIZE = 128
_BLENDER_PROCESS_POOLS: dict[int, concurrent.futures.ProcessPoolExecutor] = {}
_BLENDER_PROCESS_POOLS_LOCK = threading.Lock()


def _normalize_base_urls(
    *,
    base_url: str | None = None,
    base_urls: list[str] | None = None,
) -> list[str]:
    urls = [url for url in (base_urls or []) if str(url).strip()]
    if not urls and base_url:
        urls = [base_url]
    if not urls:
        raise ValueError("At least one Blender base URL is required.")
    return urls


def _check_blender_server_reachable(
    base_url: str,
    timeout_s: float = 3.0,
) -> None:
    root_url = base_url.rstrip("/") + "/"
    try:
        with urllib.request.urlopen(root_url, timeout=timeout_s) as response:
            status = getattr(response, "status", None)
            if status is not None and int(status) >= 400:
                raise RuntimeError(
                    f"Blender server unhealthy at {root_url}: HTTP {status}"
                )
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Failed to reach Blender server at {root_url}: {exc}"
        ) from exc


def _coerce_rgb_channels(value: Any) -> np.ndarray:
    frame = np.asarray(value)
    if frame.ndim == 4 and frame.shape[-1] == 4:
        return frame[..., :3]
    if frame.ndim == 3 and frame.shape[-1] == 4:
        return frame[..., :3]
    if frame.ndim >= 3 and frame.shape[-1] == 1:
        return np.repeat(frame, 3, axis=-1)
    return frame


def _canonicalize_render_obs_key(
    render_obs_key: RenderObsKey | None,
    camera_names: list[str],
) -> RenderObsKey:
    if render_obs_key is None:
        ordered_camera_names = [
            camera_name
            for camera_name in _DEFAULT_ALLEGRO_CAMERA_ORDER
            if camera_name in camera_names
        ]
        ordered_camera_names.extend(
            camera_name
            for camera_name in camera_names
            if camera_name not in ordered_camera_names
        )
        return [
            get_color_image_port_name_from_camera_name(camera_name)
            for camera_name in ordered_camera_names
        ]

    keys = [render_obs_key] if isinstance(render_obs_key, str) else list(render_obs_key)
    resolved: list[str] = []
    for key in keys:
        if key in camera_names:
            resolved.append(get_color_image_port_name_from_camera_name(key))
            continue
        resolved.append(key)
    if isinstance(render_obs_key, str):
        return resolved[0]
    return resolved


def _camera_names_from_render_obs_key(
    render_obs_key: RenderObsKey,
    *,
    available_camera_names: list[str],
) -> list[str]:
    keys = [render_obs_key] if isinstance(render_obs_key, str) else list(render_obs_key)
    out: list[str] = []
    for key in keys:
        if key in available_camera_names:
            out.append(key)
            continue
        prefix = "color_image_"
        if key.startswith(prefix):
            camera_name = key[len(prefix) :]
            if camera_name in available_camera_names:
                out.append(camera_name)
                continue
        raise KeyError(
            f"Expected camera name or color image key for Blender rerender, got: {key}"
        )
    return out


def _resolve_render_obs(
    obs: dict[str, Any],
    render_obs_key: RenderObsKey,
    *,
    fallback_image_keys: Optional[list[str]] = None,
) -> np.ndarray:
    keys = [render_obs_key] if isinstance(render_obs_key, str) else list(render_obs_key)
    if not keys:
        raise ValueError("render_obs_key list must contain at least one image key")

    if len(keys) == 1:
        key = keys[0]
        if key not in obs:
            if fallback_image_keys:
                fallback_key = fallback_image_keys[0]
                if fallback_key in obs:
                    key = fallback_key
                else:
                    raise KeyError(f"Fallback image key not found: {fallback_key}")
            else:
                raise KeyError(f"Render observation key not found: {key}")
        value = _coerce_rgb_channels(obs[key])
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        return np.asarray(value).copy()

    frames = []
    for key in keys:
        if key not in obs:
            raise KeyError(
                f"Render observation key not found for multiview path: {key}"
            )
        value = _coerce_rgb_channels(obs[key])
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        frames.append(np.asarray(value))

    heights = {int(frame.shape[-3]) for frame in frames}
    channels = {int(frame.shape[-1]) for frame in frames}
    if len(heights) != 1 or len(channels) != 1:
        raise ValueError(
            "All multiview render observations must share the same height and channel count"
        )

    concat_axis = 2 if frames[0].ndim == 4 else 1
    return np.concatenate(frames, axis=concat_axis).copy()


def _rgb_resize_resample() -> int:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return int(resampling.BICUBIC)
    return int(getattr(Image, "BICUBIC"))


def _resize_rgb_frame(
    value: Any,
    *,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    frame = _coerce_rgb_channels(value)
    if frame.ndim == 4:
        resized = [
            _resize_rgb_frame(
                batch_frame,
                target_height=target_height,
                target_width=target_width,
            )
            for batch_frame in frame
        ]
        return np.stack(resized, axis=0)

    frame_uint8 = np.asarray(frame)
    if frame_uint8.dtype != np.uint8:
        frame_uint8 = frame_uint8.astype(np.float32)
        if frame_uint8.size and float(np.max(frame_uint8)) <= 1.0:
            frame_uint8 = frame_uint8 * 255.0
        frame_uint8 = frame_uint8.clip(0, 255).astype(np.uint8)
    if frame_uint8.shape[:2] == (target_height, target_width):
        return frame_uint8.copy()
    return np.asarray(
        Image.fromarray(frame_uint8).resize(
            (target_width, target_height),
            _rgb_resize_resample(),
        )
    )


def _resize_selected_rgb_observations(
    obs: dict[str, Any],
    *,
    render_obs_key: RenderObsKey,
    render_size: int,
) -> dict[str, Any]:
    target_keys = (
        [render_obs_key] if isinstance(render_obs_key, str) else list(render_obs_key)
    )
    resized_obs = dict(obs)
    for key in target_keys:
        if key not in resized_obs:
            raise KeyError(f"Render observation key not found for resize: {key}")
        resized_obs[key] = _resize_rgb_frame(
            resized_obs[key],
            target_height=render_size,
            target_width=render_size,
        )
    return resized_obs


def _apply_blender_base_url_override(scenario: Any, base_url: str) -> None:
    for camera in scenario.cameras.values():
        if isinstance(camera.renderer_class, RenderEngineGltfClientParams):
            camera.renderer_class.base_url = base_url


def _get_blender_rerender_bundle(
    *,
    scenario_path: Path,
    base_url: str,
) -> dict[str, Any]:
    resolved_scenario_path = scenario_path.expanduser().resolve()
    cache_key = (str(resolved_scenario_path), base_url)
    if cache_key in _BLENDER_RERENDER_CACHE:
        return _BLENDER_RERENDER_CACHE[cache_key]

    scenario = load_scenario(filename=resolved_scenario_path)
    _apply_blender_base_url_override(scenario, base_url=base_url)

    camera_body_names = [
        camera.X_PB.base_frame.split("::")[0] for camera in scenario.cameras.values()
    ]
    station = MakeHardwareStation(
        scenario,
        meshcat=None,
        package_xmls=get_default_package_xmls(),
        hardware=False,
        visualize=False,
        disable_gravity=camera_body_names,
    )
    simulator = Simulator(station)
    simulator.Initialize()
    context = simulator.get_mutable_context()
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyMutableContextFromRoot(context)

    bundle = {
        "station": station,
        "context": context,
        "plant": plant,
        "plant_context": plant_context,
        "camera_names": [camera.name for camera in scenario.cameras.values()],
        "expected_qv_size": plant.num_positions() + plant.num_velocities(),
        "base_url": base_url,
    }
    _BLENDER_RERENDER_CACHE[cache_key] = bundle
    return bundle


def _set_blender_bundle_state_from_obs(
    bundle: dict[str, Any],
    obs: dict[str, Any],
) -> np.ndarray:
    qv = np.asarray(obs["qv"]).reshape((-1,))
    if qv.size != bundle["expected_qv_size"]:
        raise ValueError(
            f"Bad qv size: got {qv.size}, expected {bundle['expected_qv_size']}"
        )

    if "time" in obs:
        bundle["context"].SetTime(float(obs["time"]))
    bundle["plant"].SetPositionsAndVelocities(
        bundle["plant_context"],
        qv,
    )
    return qv


def _rerender_camera_batch_on_url(
    *,
    obs: dict[str, Any],
    scenario_path: Path,
    base_url: str,
    camera_names: list[str],
) -> dict[str, np.ndarray]:
    _check_blender_server_reachable(base_url)
    bundle = _get_blender_rerender_bundle(
        scenario_path=scenario_path,
        base_url=base_url,
    )
    _set_blender_bundle_state_from_obs(bundle, obs)

    rerendered_images: dict[str, np.ndarray] = {}
    for camera_name in camera_names:
        color_port_name = get_color_image_port_name_from_camera_name(camera_name)
        try:
            rgba = (
                bundle["station"]
                .GetOutputPort(color_port_name)
                .Eval(bundle["context"])
                .data
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Blender RGB rerender failed for "
                f"{camera_name} via {base_url}. "
                "The Blender server is reachable but could not render the scene. "
                "If the error mentions `bpy.ops.import_scene.gltf` or missing "
                "Blender operators, restart the render servers from a working "
                "drake-blender checkout with the expected settings file."
            ) from exc
        rerendered_images[color_port_name] = np.array(rgba, copy=True)
    return rerendered_images


def _candidate_blender_urls(
    *,
    urls: list[str],
    preferred_url: str,
) -> list[str]:
    ordered_urls = [preferred_url]
    ordered_urls.extend(url for url in urls if url != preferred_url)
    return ordered_urls


def _camera_batches_for_urls(
    *,
    requested_camera_names: list[str],
    urls: list[str],
    url_index: int,
) -> collections.OrderedDict[str, list[str]]:
    camera_batches_by_url: collections.OrderedDict[str, list[str]] = (
        collections.OrderedDict()
    )
    for camera_offset, camera_name in enumerate(requested_camera_names):
        assigned_url = urls[(url_index + camera_offset) % len(urls)]
        camera_batches_by_url.setdefault(assigned_url, []).append(camera_name)
    return camera_batches_by_url


def _rerender_camera_batch_with_failover(
    *,
    obs: dict[str, Any],
    scenario_path: Path,
    candidate_urls: list[str],
    camera_names: list[str],
) -> dict[str, np.ndarray]:
    render_errors: list[tuple[str, Exception]] = []
    for candidate_url in candidate_urls:
        try:
            return _rerender_camera_batch_on_url(
                obs=obs,
                scenario_path=scenario_path,
                base_url=candidate_url,
                camera_names=camera_names,
            )
        except Exception as exc:
            _BLENDER_RERENDER_CACHE.clear()
            render_errors.append((candidate_url, exc))
            if len(candidate_urls) > 1:
                warnings.warn(
                    "Blender rerender via "
                    f"{candidate_url} failed for cameras {camera_names}; "
                    "retrying on another server.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    tried_urls = [url for url, _ in render_errors]
    last_url, last_error = render_errors[-1]
    raise RuntimeError(
        "Blender RGB rerender failed after trying "
        f"{tried_urls} for cameras {camera_names}. "
        f"Last failing URL: {last_url}."
    ) from last_error


def _rerender_camera_batch_with_failover_subprocess(
    *,
    obs: dict[str, Any],
    scenario_path: str,
    candidate_urls: list[str],
    camera_names: list[str],
) -> dict[str, np.ndarray]:
    return _rerender_camera_batch_with_failover(
        obs=obs,
        scenario_path=Path(scenario_path),
        candidate_urls=candidate_urls,
        camera_names=camera_names,
    )


def _rerender_camera_batches_sequentially(
    *,
    obs: dict[str, Any],
    scenario_path: Path,
    camera_batches_by_url: collections.OrderedDict[str, list[str]],
    urls: list[str],
) -> dict[str, np.ndarray]:
    rerendered_images: dict[str, np.ndarray] = {}
    for assigned_url, assigned_camera_names in camera_batches_by_url.items():
        rerendered_images.update(
            _rerender_camera_batch_with_failover(
                obs=obs,
                scenario_path=scenario_path,
                candidate_urls=_candidate_blender_urls(
                    urls=urls,
                    preferred_url=assigned_url,
                ),
                camera_names=assigned_camera_names,
            )
        )
    return rerendered_images


def _multiview_parallel_backend() -> str:
    backend = os.environ.get("ALLEGRO_BLENDER_MULTIVIEW_BACKEND", "process")
    normalized = backend.strip().lower()
    if normalized not in {"process", "thread", "sequential"}:
        warnings.warn(
            "Unknown ALLEGRO_BLENDER_MULTIVIEW_BACKEND="
            f"{backend!r}; falling back to 'process'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "process"
    return normalized


def _get_blender_process_pool(
    worker_count: int,
) -> concurrent.futures.ProcessPoolExecutor:
    with _BLENDER_PROCESS_POOLS_LOCK:
        pool = _BLENDER_PROCESS_POOLS.get(worker_count)
        if pool is None:
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("spawn"),
            )
            _BLENDER_PROCESS_POOLS[worker_count] = pool
        return pool


def _shutdown_blender_process_pools() -> None:
    with _BLENDER_PROCESS_POOLS_LOCK:
        pools = list(_BLENDER_PROCESS_POOLS.values())
        _BLENDER_PROCESS_POOLS.clear()
    for pool in pools:
        pool.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_blender_process_pools)


def rerender_rgbs_in_blender(
    obs: dict[str, Any],
    *,
    render_obs_key: RenderObsKey,
    scenario_path: Path,
    base_url: str | None = None,
    base_urls: list[str] | None = None,
    camera_names: list[str] | None = None,
    url_index: int = 0,
) -> dict[str, Any]:
    urls = _normalize_base_urls(base_url=base_url, base_urls=base_urls)
    resolved_base_url = urls[url_index % len(urls)]
    _check_blender_server_reachable(resolved_base_url)
    bundle = _get_blender_rerender_bundle(
        scenario_path=scenario_path,
        base_url=resolved_base_url,
    )
    requested_camera_names = (
        camera_names
        if camera_names is not None
        else _camera_names_from_render_obs_key(
            render_obs_key,
            available_camera_names=bundle["camera_names"],
        )
    )

    rerendered_obs = dict(obs)
    if len(urls) == 1 or len(requested_camera_names) <= 1:
        rerendered_obs.update(
            _rerender_camera_batch_with_failover(
                obs=obs,
                scenario_path=scenario_path,
                candidate_urls=_candidate_blender_urls(
                    urls=urls,
                    preferred_url=resolved_base_url,
                ),
                camera_names=requested_camera_names,
            )
        )
        return rerendered_obs

    camera_batches_by_url = _camera_batches_for_urls(
        requested_camera_names=requested_camera_names,
        urls=urls,
        url_index=url_index,
    )
    worker_count = min(len(camera_batches_by_url), len(requested_camera_names))
    parallel_backend = _multiview_parallel_backend()
    if parallel_backend == "sequential":
        rerendered_obs.update(
            _rerender_camera_batches_sequentially(
                obs=obs,
                scenario_path=scenario_path,
                camera_batches_by_url=camera_batches_by_url,
                urls=urls,
            )
        )
        return rerendered_obs
    try:
        if parallel_backend == "thread":
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count
            ) as pool:
                future_to_url = {
                    pool.submit(
                        _rerender_camera_batch_with_failover,
                        obs=obs,
                        scenario_path=scenario_path,
                        candidate_urls=_candidate_blender_urls(
                            urls=urls,
                            preferred_url=assigned_url,
                        ),
                        camera_names=assigned_camera_names,
                    ): assigned_url
                    for assigned_url, assigned_camera_names in camera_batches_by_url.items()
                }
                for future in concurrent.futures.as_completed(future_to_url):
                    rerendered_obs.update(future.result())
        else:
            pool = _get_blender_process_pool(worker_count)
            future_to_url = {
                pool.submit(
                    _rerender_camera_batch_with_failover_subprocess,
                    obs=obs,
                    scenario_path=str(scenario_path),
                    candidate_urls=_candidate_blender_urls(
                        urls=urls,
                        preferred_url=assigned_url,
                    ),
                    camera_names=assigned_camera_names,
                ): assigned_url
                for assigned_url, assigned_camera_names in camera_batches_by_url.items()
            }
            for future in concurrent.futures.as_completed(future_to_url):
                rerendered_obs.update(future.result())
    except Exception as exc:
        # GPU-backed drake-blender can intermittently fail when multiple render
        # requests race at the same time. Clear the cached rerender bundles and
        # retry the same fanout sequentially instead of aborting the notebook.
        _BLENDER_RERENDER_CACHE.clear()
        warnings.warn(
            "Parallel multiview Blender rerender "
            f"(backend={parallel_backend}) failed; retrying sequentially. "
            f"Original error: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        rerendered_obs.update(
            _rerender_camera_batches_sequentially(
                obs=obs,
                scenario_path=scenario_path,
                camera_batches_by_url=camera_batches_by_url,
                urls=urls,
            )
        )
    return rerendered_obs


def rerender_obs_sequence_in_blender(
    obs_sequence: list[dict[str, Any]],
    *,
    render_obs_key: RenderObsKey,
    scenario_path: Path,
    base_url: str | None = None,
    base_urls: list[str] | None = None,
    camera_names: list[str] | None = None,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    urls = _normalize_base_urls(base_url=base_url, base_urls=base_urls)
    if len(urls) == 1 or len(obs_sequence) <= 1:
        return [
            rerender_rgbs_in_blender(
                obs,
                render_obs_key=render_obs_key,
                scenario_path=scenario_path,
                base_urls=urls,
                camera_names=camera_names,
                url_index=index,
            )
            for index, obs in enumerate(obs_sequence)
        ]

    worker_count = max_workers or min(len(urls), len(obs_sequence))
    results: list[dict[str, Any] | None] = [None] * len(obs_sequence)
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_to_index = {
            pool.submit(
                rerender_rgbs_in_blender,
                obs,
                render_obs_key=render_obs_key,
                scenario_path=scenario_path,
                base_urls=urls,
                camera_names=camera_names,
                url_index=index,
            ): index
            for index, obs in enumerate(obs_sequence)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    return [result for result in results if result is not None]


@dataclass
class AllegroRunnerCfg(BaseRunnerCfg):
    env_name: Literal["allegro"]

    hydra_cfg_path: str = str(
        NEURAL_JACOBIAN_DIR / "allegro" / "config" / "example_relative_yaw90.yaml"
    )
    drake_scenario_path: str = str(
        NEURAL_JACOBIAN_DIR / "scenarios" / "allegro_hand_and_box_camera.scenario.yaml"
    )
    blender_scenario_path: str = str(
        NEURAL_JACOBIAN_DIR
        / "scenarios"
        / "allegro_hand_and_box_camera_blender.scenario.yaml"
    )
    use_blender_rgb: bool = True
    render_obs_key: RenderObsKey | None = None
    render_size: int = _DEFAULT_ALLEGRO_RENDER_SIZE
    blender_base_url: str = "http://127.0.0.1:8000"
    blender_base_urls: list[str] | None = None
    scale_blender_scenario_to_render_size: bool = False
    scaled_blender_scenario_output_dir: str | None = None

    num_envs: int = 1
    seed: int = 0
    max_episode_steps: int = 20
    env_n_action_steps: int | None = None
    env_n_obs_steps: int | None = None
    n_repeat: int = 1
    action_scale: float = 1.0

    output_dir: str = "outputs/allegro_eval"
    save_videos: bool = True
    save_trajectory: bool = True
    save_rrd: bool = True
    video_fps: int = 10

    visualize: bool = True
    add_logger: bool = False

    # The motion policy emits joint-space deltas; the runner integrates them
    # against a persistent filtered joint reference before sending absolute
    # targets. The live observed state is still exposed to the policy.
    action_mode: Literal["velocity", "absolute"] = "velocity"
    dt: float = 0.1
    log_step_debug: bool = True
    q_a_ref_update_every: int = 1
    q_a_ref_obs_alpha: float = 0.2
    q_a_ref_target_alpha: float = 1.0

    q_cameras: dict[str, list[float]] | None = None
    q_initial: list[float] | None = None
    q_u_goal: list[float] | None = None

    # Override env.goal_threshold from the loaded hydra cfg. None = leave as-is.
    # Distance metric is 0.5*angular_rad + 0.5*translation_m, so 0.06 ≈ 6.9°
    # for a pure-rotation goal (very strict). Loosen for paper SR if needed.
    env_goal_threshold: float | None = None

    def resolved_blender_base_urls(self) -> list[str]:
        return _normalize_base_urls(
            base_url=self.blender_base_url,
            base_urls=self.blender_base_urls,
        )


def _normalize_obs_for_policy(
    obs: dict[str, Any], shape_meta: dict[str, Any]
) -> dict[str, Any]:
    out = {}
    for key, spec in shape_meta["obs"].items():
        if key not in obs:
            continue
        value = np.asarray(obs[key])
        if spec.get("type") == "rgb":
            value = _coerce_rgb_channels(value)
            if value.dtype == np.uint8:
                value = value.astype(np.float32) / 255.0
            else:
                value = value.astype(np.float32)
                if value.size and float(np.max(value)) > 1.0:
                    value = value / 255.0
        else:
            value = value.astype(np.float32)
        out[key] = value
    return out


class AllegroImageWrapper(gym.Env):
    def __init__(
        self,
        env,
        *,
        shape_meta: dict[str, Any],
        render_obs_key: RenderObsKey,
        blender_scenario_path: Path,
        use_blender_rgb: bool = True,
        render_size: int = _DEFAULT_ALLEGRO_RENDER_SIZE,
        blender_base_url: str | None = None,
        blender_base_urls: list[str] | None = None,
    ):
        super().__init__()
        self.env = env
        self.shape_meta = shape_meta
        self.render_obs_key = render_obs_key
        self.blender_scenario_path = blender_scenario_path
        self.use_blender_rgb = use_blender_rgb
        self.render_size = int(render_size)
        self.blender_base_url = blender_base_url
        self.blender_base_urls = blender_base_urls
        self.render_cache = None
        self.latest_raw_obs = None
        self._render_url_counter = 0
        self.latest_observation_timing: dict[str, float] = {}
        self.latest_timing: dict[str, float] = {}

        self.action_space = env.action_space
        observation_space = gym.spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = tuple(int(x) for x in value["shape"])
            low, high = (0, 1) if value.get("type") == "rgb" else (-np.inf, np.inf)
            observation_space[key] = gym.spaces.Box(
                low=low,
                high=high,
                shape=shape,
                dtype=np.float32,
            )
        self.observation_space = observation_space

    def _next_render_url_index(self) -> int:
        self._render_url_counter += 1
        return next(_BLENDER_URL_COUNTER)

    def get_observation(self, raw_obs_list=None) -> dict[str, Any]:
        if raw_obs_list is None:
            raise ValueError("raw_obs_list must be provided for AllegroImageWrapper.")
        raw_obs = raw_obs_list[-1] if isinstance(raw_obs_list, list) else raw_obs_list
        self.latest_raw_obs = raw_obs

        observation_t0 = time.perf_counter()
        display_obs = raw_obs
        blender_rerender_s = 0.0
        if self.use_blender_rgb:
            rerender_t0 = time.perf_counter()
            display_obs = rerender_rgbs_in_blender(
                raw_obs,
                render_obs_key=self.render_obs_key,
                scenario_path=self.blender_scenario_path,
                base_url=self.blender_base_url,
                base_urls=self.blender_base_urls,
                url_index=self._next_render_url_index(),
            )
            blender_rerender_s = time.perf_counter() - rerender_t0
        resize_t0 = time.perf_counter()
        display_obs = _resize_selected_rgb_observations(
            display_obs,
            render_obs_key=self.render_obs_key,
            render_size=self.render_size,
        )
        resize_s = time.perf_counter() - resize_t0

        normalize_t0 = time.perf_counter()
        self.render_cache = _resolve_render_obs(display_obs, self.render_obs_key)
        normalized_obs = _normalize_obs_for_policy(display_obs, self.shape_meta)
        normalize_s = time.perf_counter() - normalize_t0
        self.latest_observation_timing = {
            "blender_rerender_s": float(blender_rerender_s),
            "resize_s": float(resize_s),
            "normalize_s": float(normalize_s),
            "observation_total_s": float(time.perf_counter() - observation_t0),
        }
        return normalized_obs

    def reset(self, seed=None, options=None):
        kwargs = dict(options or {})
        reset_t0 = time.perf_counter()
        raw_reset_t0 = time.perf_counter()
        obs_list, info = self.env.reset(seed=seed, **kwargs)
        raw_env_reset_s = time.perf_counter() - raw_reset_t0
        obs = self.get_observation(obs_list)
        self.latest_timing = {
            "raw_env_reset_s": float(raw_env_reset_s),
            **self.latest_observation_timing,
            "wrapper_reset_total_s": float(time.perf_counter() - reset_t0),
        }
        return obs, info

    def step(self, action):
        action_np = np.asarray(action)
        step_t0 = time.perf_counter()
        raw_step_t0 = time.perf_counter()
        obs_list, reward, terminated, truncated, info = self.env.step(action_np)
        raw_env_step_s = time.perf_counter() - raw_step_t0
        obs = self.get_observation(obs_list)
        self.latest_timing = {
            "raw_env_step_s": float(raw_env_step_s),
            **self.latest_observation_timing,
            "wrapper_step_total_s": float(time.perf_counter() - step_t0),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_cache is None:
            raise RuntimeError("Must call reset or step before render().")
        return np.asarray(self.render_cache).copy()

    def close(self):
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            close_fn()


class AllegroRunner(BaseRunner):
    cfg: AllegroRunnerCfg

    def __init__(
        self,
        cfg: AllegroRunnerCfg,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super().__init__(cfg, device)

    def _step_debug_enabled(self) -> bool:
        return bool(getattr(self.cfg, "log_step_debug", True))

    def _step_debug_write(self, message: str) -> None:
        if self._step_debug_enabled():
            tqdm.tqdm.write(message)

    @staticmethod
    def _flatten_info_values(value: Any) -> list[Any]:
        if isinstance(value, np.ndarray):
            return [
                item
                for element in np.asarray(value, dtype=object).reshape(-1).tolist()
                for item in AllegroRunner._flatten_info_values(element)
            ]
        if isinstance(value, (list, tuple)):
            return [item for element in value for item in AllegroRunner._flatten_info_values(element)]
        return [value]

    @classmethod
    def _extract_info_field_values(cls, info: Any, key: str) -> list[Any]:
        if not isinstance(info, dict):
            return []
        direct_values = cls._flatten_info_values(info.get(key))
        if direct_values:
            return direct_values
        final_info = info.get("final_info")
        extracted: list[Any] = []
        for item in cls._flatten_info_values(final_info):
            if isinstance(item, dict) and key in item:
                extracted.extend(cls._flatten_info_values(item.get(key)))
        return extracted

    @staticmethod
    def _format_compact_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

    @classmethod
    def _format_info_field(cls, info: Any, key: str) -> str:
        values = cls._extract_info_field_values(info, key)
        if not values:
            return "n/a"
        compact = [cls._format_compact_value(value) for value in values]
        if len(compact) == 1:
            return str(compact[0])
        return str(compact)

    @classmethod
    def _rollout_status_line(
        cls,
        *,
        step_index: int | None,
        reward: np.ndarray | None,
        terminated: np.ndarray | None,
        truncated: np.ndarray | None,
        info: Any,
    ) -> str:
        reward_text = "n/a" if reward is None else str(np.asarray(reward).tolist())
        terminated_text = (
            "n/a" if terminated is None else str(np.asarray(terminated, dtype=bool).tolist())
        )
        truncated_text = (
            "n/a" if truncated is None else str(np.asarray(truncated, dtype=bool).tolist())
        )
        if terminated is None or truncated is None:
            done_text = "n/a"
        else:
            done_text = str(
                np.logical_or(
                    np.asarray(terminated, dtype=bool),
                    np.asarray(truncated, dtype=bool),
                ).tolist()
            )
        return (
            f"step={step_index} reward={reward_text} done={done_text} "
            f"terminated={terminated_text} truncated={truncated_text} "
            f"goal={cls._format_info_field(info, 'has_reached_goal')} "
            f"bounds={cls._format_info_field(info, 'q_u_within_bounds')} "
            f"timeout={cls._format_info_field(info, 'time_limit_exceeded')}"
        )

    @staticmethod
    def _unwrap_env_attr(env: Any, attr_name: str) -> Any:
        current = env
        while current is not None:
            if hasattr(current, attr_name):
                return getattr(current, attr_name)
            current = getattr(current, "env", None)
        return None

    def _latest_env_timing_summary(self) -> dict[str, float]:
        if not hasattr(self, "env") or not hasattr(self.env, "envs"):
            return {}
        timing_keys = [
            "raw_env_step_s",
            "blender_rerender_s",
            "resize_s",
            "normalize_s",
            "observation_total_s",
            "wrapper_step_total_s",
        ]
        collected: dict[str, list[float]] = {key: [] for key in timing_keys}
        for env in self.env.envs:
            latest_timing = self._unwrap_env_attr(env, "latest_timing")
            if not isinstance(latest_timing, dict):
                continue
            for key in timing_keys:
                value = latest_timing.get(key)
                if value is None:
                    continue
                collected[key].append(float(value))
        return {
            key: float(np.mean(values))
            for key, values in collected.items()
            if values
        }

    @staticmethod
    def _sanitize_run_tag(tag: str) -> str:
        safe = []
        for ch in tag.strip():
            if ch.isalnum() or ch in ("-", "_", "."):
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe).strip("._-") or "run"

    def _resolved_q_cameras(self) -> dict[str, np.ndarray]:
        if self.cfg.q_cameras is None:
            return get_default_q_cameras()
        return {
            key: np.asarray(value, dtype=np.float64)
            for key, value in self.cfg.q_cameras.items()
        }

    def _infer_shape_meta(
        self,
        sample_obs: dict[str, Any],
        *,
        action_space: gym.Space,
    ) -> dict[str, Any]:
        obs_shapes = {}
        for key, value in sample_obs.items():
            arr = np.asarray(value)
            if key.startswith("color_image_"):
                arr = _coerce_rgb_channels(arr)
                obs_type = "rgb"
            else:
                obs_type = "low_dim"
            obs_shapes[key] = {
                "shape": list(arr.shape),
                "type": obs_type,
            }
        action_shape = list(getattr(action_space, "shape", ()))
        return {"obs": obs_shapes, "action": {"shape": action_shape}}

    def setup_env(self):
        if_docker_then_check_display_and_x_server()
        register_resolvers()

        self.base_blender_scenario_path = Path(self.cfg.blender_scenario_path).resolve()
        self.effective_blender_scenario_path = maybe_make_scaled_camera_scenario(
            base_path=self.base_blender_scenario_path,
            target_width=int(self.cfg.render_size),
            target_height=int(self.cfg.render_size),
            enabled=bool(self.cfg.scale_blender_scenario_to_render_size),
            output_dir=self.cfg.scaled_blender_scenario_output_dir,
            scenario_label="blender",
        )

        base_cfg = load_hydra_cfg(path=Path(self.cfg.hydra_cfg_path))
        base_cfg.env.visualize = bool(self.cfg.visualize)
        base_cfg.env.add_logger = bool(self.cfg.add_logger)
        base_cfg.env.env_scenario_path = str(Path(self.cfg.drake_scenario_path))
        if self.cfg.env_n_action_steps is not None:
            base_cfg.env.n_action_steps = int(self.cfg.env_n_action_steps)
        if self.cfg.env_n_obs_steps is not None:
            base_cfg.env.n_obs_steps = int(self.cfg.env_n_obs_steps)
        if self.cfg.env_goal_threshold is not None:
            base_cfg.env.goal_threshold = float(self.cfg.env_goal_threshold)

        prototype_env = make_env(base_cfg)
        try:
            sample_obs_list, _ = prototype_env.reset(
                q_cameras=self._resolved_q_cameras(),
                q_initial=(
                    None
                    if self.cfg.q_initial is None
                    else np.asarray(self.cfg.q_initial, dtype=np.float64)
                ),
                q_u_goal=(
                    None
                    if self.cfg.q_u_goal is None
                    else np.asarray(self.cfg.q_u_goal, dtype=np.float64)
                ),
                seed=int(self.cfg.seed),
            )
            sample_obs = sample_obs_list[-1]
            self.camera_names = list(prototype_env.all_camera_data.keys())
            self.shape_meta = self._infer_shape_meta(
                sample_obs,
                action_space=prototype_env.action_space,
            )
            q_sim = getattr(prototype_env, "q_sim", None)
            if q_sim is None:
                raise AttributeError("Allegro prototype env is missing `q_sim`.")
            self.q_a_indices = np.asarray(
                q_sim.get_q_a_indices_into_q(),
                dtype=np.int64,
            )
            action_space = prototype_env.action_space
            action_low = np.asarray(getattr(action_space, "low"), dtype=np.float32)
            action_high = np.asarray(getattr(action_space, "high"), dtype=np.float32)
            self.q_a_lower_limits = np.asarray(action_low[0], dtype=np.float32)
            self.q_a_upper_limits = np.asarray(action_high[0], dtype=np.float32)
        finally:
            close_fn = getattr(prototype_env, "close", None)
            if callable(close_fn):
                close_fn()

        self.render_obs_key = _canonicalize_render_obs_key(
            self.cfg.render_obs_key,
            self.camera_names,
        )
        resized_render_keys = (
            [self.render_obs_key]
            if isinstance(self.render_obs_key, str)
            else list(self.render_obs_key)
        )
        for key in resized_render_keys:
            if key not in self.shape_meta["obs"]:
                continue
            self.shape_meta["obs"][key]["shape"] = [
                int(self.cfg.render_size),
                int(self.cfg.render_size),
                3,
            ]
            self.shape_meta["obs"][key]["type"] = "rgb"
        self.image_keys = [
            key
            for key, spec in self.shape_meta["obs"].items()
            if spec.get("type") == "rgb"
        ]
        self.blender_base_urls = self.cfg.resolved_blender_base_urls()
        self.env_cfg = base_cfg

        def env_fn():
            env_cfg = copy.deepcopy(base_cfg)
            raw_env = make_env(env_cfg)
            wrapped = AllegroImageWrapper(
                raw_env,
                shape_meta=self.shape_meta,
                render_obs_key=self.render_obs_key,
                blender_scenario_path=self.effective_blender_scenario_path,
                use_blender_rgb=bool(self.cfg.use_blender_rgb),
                render_size=int(self.cfg.render_size),
                blender_base_urls=self.blender_base_urls,
            )
            wrapped = gym.wrappers.PassiveEnvChecker(wrapped)
            wrapped = gym.wrappers.OrderEnforcing(wrapped)
            wrapped = gym.wrappers.TimeLimit(
                wrapped,
                max_episode_steps=int(self.cfg.max_episode_steps),
            )
            return wrapped

        num_envs = max(1, int(self.cfg.num_envs))
        self.env = gym.vector.SyncVectorEnv([env_fn for _ in range(num_envs)])
        self._num_envs = num_envs

    def close(self) -> None:
        close_fn = getattr(getattr(self, "env", None), "close", None)
        if callable(close_fn):
            close_fn()

    def _policy_rgb_from_obs(self, obs: dict[str, Any]) -> np.ndarray:
        return _resolve_render_obs(
            obs,
            self.render_obs_key,
            fallback_image_keys=self.image_keys,
        )

    @staticmethod
    def _frame_width(frame: Any) -> int:
        value = np.asarray(frame)
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        return int(value.shape[-2])

    def _policy_view_metadata(
        self,
        obs: dict[str, Any],
    ) -> tuple[list[str], list[int], str]:
        if isinstance(self.render_obs_key, str):
            key = self.render_obs_key
            if key not in obs:
                raise KeyError(f"Render observation key not found: {key}")
            return [key], [self._frame_width(obs[key])], key

        view_keys = list(self.render_obs_key)
        if not view_keys:
            raise ValueError("render_obs_key list must contain at least one key")
        for key in view_keys:
            if key not in obs:
                raise KeyError(
                    f"Render observation key not found for multiview path: {key}"
                )
        view_widths = [self._frame_width(obs[key]) for key in view_keys]
        return view_keys, view_widths, "|".join(view_keys)

    @staticmethod
    def _batched_obs_value(obs: dict[str, Any], key: str) -> np.ndarray | None:
        value = obs.get(key)
        if value is None:
            return None
        value_np = np.asarray(value)
        if value_np.ndim == 1:
            value_np = value_np[None, :]
        return value_np

    def _current_q_a_from_obs(self, obs: dict[str, Any]) -> np.ndarray:
        q = self._batched_obs_value(obs, "q")
        if q is None:
            raise KeyError("Allegro observation is missing `q`.")

        q_a_indices = getattr(self, "q_a_indices", None)
        if q_a_indices is None or q.shape[-1] == len(q_a_indices):
            return np.asarray(q, dtype=np.float32)
        if q.shape[-1] <= int(np.max(q_a_indices)):
            raise ValueError(
                "Allegro observation `q` is too short to extract actuated joints: "
                f"q shape={q.shape}, q_a_indices={q_a_indices.tolist()}"
            )
        return np.asarray(q[..., q_a_indices], dtype=np.float32)

    def _clip_q_a_targets(self, q_a_targets: np.ndarray) -> np.ndarray:
        lower = getattr(self, "q_a_lower_limits", None)
        upper = getattr(self, "q_a_upper_limits", None)
        if lower is None or upper is None:
            return np.asarray(q_a_targets, dtype=np.float32)
        return np.clip(
            np.asarray(q_a_targets, dtype=np.float32),
            lower.reshape((1,) * (q_a_targets.ndim - 1) + (-1,)),
            upper.reshape((1,) * (q_a_targets.ndim - 1) + (-1,)),
        )

    @staticmethod
    def _match_q_a_batch(
        q_a_value: np.ndarray,
        *,
        batch_size: int,
        value_name: str,
    ) -> np.ndarray:
        q_a_value = np.asarray(q_a_value, dtype=np.float32)
        if q_a_value.ndim == 1:
            q_a_value = q_a_value[None, :]
        if q_a_value.ndim != 2:
            raise ValueError(
                f"Expected {value_name} with shape (B, q_a_dim), got {q_a_value.shape}"
            )
        if q_a_value.shape[0] == batch_size:
            return q_a_value
        if q_a_value.shape[0] == 1 and batch_size > 1:
            return np.repeat(q_a_value, repeats=batch_size, axis=0)
        raise ValueError(
            f"Allegro {value_name} batch does not match target batch size: "
            f"{q_a_value.shape} vs {batch_size}"
        )

    @staticmethod
    def _blend_q_a_reference(
        q_a_ref: np.ndarray,
        q_a_source: np.ndarray,
        *,
        alpha: float,
    ) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return np.asarray(q_a_ref, dtype=np.float32)
        if alpha >= 1.0:
            return np.asarray(q_a_source, dtype=np.float32)
        return np.asarray(
            (1.0 - alpha) * np.asarray(q_a_ref, dtype=np.float32)
            + alpha * np.asarray(q_a_source, dtype=np.float32),
            dtype=np.float32,
        )

    def _update_q_a_reference_from_observation(
        self,
        q_a_ref: np.ndarray,
        q_a_measured: np.ndarray,
        *,
        step_index: int,
    ) -> np.ndarray:
        ref_batch_size = 1 if np.asarray(q_a_ref).ndim == 1 else int(q_a_ref.shape[0])
        q_a_ref = self._match_q_a_batch(
            q_a_ref,
            batch_size=max(1, ref_batch_size),
            value_name="reference state",
        )
        update_every = max(1, int(getattr(self.cfg, "q_a_ref_update_every", 1)))
        if step_index % update_every != 0:
            return self._clip_q_a_targets(q_a_ref)
        q_a_measured = self._match_q_a_batch(
            q_a_measured,
            batch_size=q_a_ref.shape[0],
            value_name="measured joint state",
        )
        return self._clip_q_a_targets(
            self._blend_q_a_reference(
                q_a_ref,
                q_a_measured,
                alpha=float(getattr(self.cfg, "q_a_ref_obs_alpha", 1.0)),
            )
        )

    def _update_q_a_reference_from_target(
        self,
        q_a_ref: np.ndarray,
        q_a_target: np.ndarray,
    ) -> np.ndarray:
        ref_batch_size = 1 if np.asarray(q_a_ref).ndim == 1 else int(q_a_ref.shape[0])
        q_a_ref = self._match_q_a_batch(
            q_a_ref,
            batch_size=max(1, ref_batch_size),
            value_name="reference state",
        )
        q_a_target = self._match_q_a_batch(
            q_a_target,
            batch_size=q_a_ref.shape[0],
            value_name="target joint state",
        )
        return self._clip_q_a_targets(
            self._blend_q_a_reference(
                q_a_ref,
                q_a_target,
                alpha=float(getattr(self.cfg, "q_a_ref_target_alpha", 1.0)),
            )
        )

    def _policy_action_to_env_target(
        self,
        action: np.ndarray,
        *,
        q_a_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 2:
            raise ValueError(
                "Expected Allegro action with shape (B, q_a_dim), "
                f"got {action.shape}"
            )

        if self.cfg.action_mode == "absolute":
            return self._clip_q_a_targets(action)

        if q_a_ref is None:
            raise ValueError("Relative Allegro actions require an explicit q_a_ref.")
        q_a_ref = self._match_q_a_batch(
            q_a_ref,
            batch_size=action.shape[0],
            value_name="reference state",
        )

        # The policy emits joint-space deltas; the runner integrates them
        # against a held/filtered q_a_ref before sending absolute targets.
        return self._clip_q_a_targets(q_a_ref + action * float(self.cfg.action_scale))

    def _policy_state_observation(
        self,
        obs: dict[str, Any],
        *,
        step_index: int | None = 0,
    ) -> PolicyObservation:
        rgb = self._policy_rgb_from_obs(obs)
        if rgb.ndim == 3:
            rgb = rgb[None, ...]
        view_keys, view_widths, concat_rgb_key = self._policy_view_metadata(obs)
        return PolicyObservation(
            rgb=rgb,
            q_robot=self._current_q_a_from_obs(obs),
            rgb_vis=rgb.copy(),
            view_keys=view_keys,
            view_widths=view_widths,
            concat_rgb_key=concat_rgb_key,
            step_index=step_index,
            eef_pos=None,
            eef_quat=None,
            gripper_qpos=None,
            dt=self.cfg.dt,
            action_mode=self.cfg.action_mode,
        )

    @staticmethod
    def _to_uint8(frame: np.ndarray) -> np.ndarray:
        value = np.asarray(frame)
        if value.dtype == np.uint8:
            return value
        value = value.astype(np.float32)
        if value.size and float(np.max(value)) <= 1.0:
            value = value * 255.0
        return value.clip(0, 255).astype(np.uint8)

    @staticmethod
    def _policy_vis_frame(policy_vis: Any) -> np.ndarray:
        vis_frame = np.asarray(policy_vis)
        if vis_frame.ndim == 4:
            vis_frame = vis_frame[0]
        return vis_frame

    def _save_video(self, frames: list[np.ndarray], path: Path) -> None:
        if not frames:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(
            str(path),
            [self._to_uint8(frame) for frame in frames],
            fps=self.cfg.video_fps,
        )

    def _default_reset_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "q_cameras": self._resolved_q_cameras(),
        }
        if self.cfg.q_initial is not None:
            options["q_initial"] = np.asarray(self.cfg.q_initial, dtype=np.float64)
        if self.cfg.q_u_goal is not None:
            options["q_u_goal"] = np.asarray(self.cfg.q_u_goal, dtype=np.float64)
        return options

    def run(
        self,
        policy: BasePolicy,
        options=None,
        run_tag: str | None = None,
        post_reset_callback: Callable[[dict[str, Any], Any, dict[str, Any]], Any]
        | None = None,
    ):
        n_envs = int(self._num_envs)
        seeds = [int(self.cfg.seed) + idx for idx in range(n_envs)]
        reset_options = self._default_reset_options()
        if options:
            reset_options.update(options)

        obs, info = self.env.reset(
            seed=seeds if n_envs > 1 else seeds[0],
            options=reset_options,
        )
        if callable(post_reset_callback):
            callback_result = post_reset_callback(obs, info, reset_options)
            if isinstance(callback_result, tuple):
                if len(callback_result) != 2:
                    raise ValueError(
                        "post_reset_callback must return either obs or (obs, info)."
                    )
                obs, info = callback_result
            elif callback_result is not None:
                obs = callback_result
        del info
        policy.reset()
        q_a_ref = self._clip_q_a_targets(self._current_q_a_from_obs(obs).copy())

        max_steps = int(self.cfg.max_episode_steps)
        episode_rewards = np.zeros(n_envs, dtype=np.float32)
        max_rewards = np.full(n_envs, -np.inf, dtype=np.float32)
        done_flags = np.zeros(n_envs, dtype=bool)
        last_reward: np.ndarray | None = None
        last_terminated: np.ndarray | None = None
        last_truncated: np.ndarray | None = None
        last_info: Any = None
        last_step_index: int | None = None
        timing_history: dict[str, list[float]] = {
            "policy_inference_s": [],
            "raw_env_step_s": [],
            "blender_rerender_s": [],
            "resize_s": [],
            "normalize_s": [],
            "observation_total_s": [],
            "wrapper_step_total_s": [],
        }

        videos = {
            "obs": [self._policy_rgb_from_obs(obs)],
            "policy": [],
        }
        traj = {
            "obs": [],
            "actions": [],
            "q_robot": [],
            "q_reference": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
        }

        ran = tqdm.trange(max_steps, desc="[AllegroRunner] Rollout")
        for step_index in ran:
            if np.all(done_flags):
                print(
                    cyan(
                        "[AllegroRunner] All envs done, stopping rollout. "
                        + self._rollout_status_line(
                            step_index=last_step_index,
                            reward=last_reward,
                            terminated=last_terminated,
                            truncated=last_truncated,
                            info=last_info,
                        )
                    )
                )
                break

            policy_obs = self._policy_state_observation(obs, step_index=step_index)
            if self.cfg.action_mode != "absolute":
                q_a_ref = self._update_q_a_reference_from_observation(
                    q_a_ref,
                    self._current_q_a_from_obs(obs),
                    step_index=step_index,
                )
            policy_t0 = time.perf_counter()
            policy_out: PolicyOutput = policy.predict_action(policy_obs)
            policy_inference_s = time.perf_counter() - policy_t0
            timing_history["policy_inference_s"].append(float(policy_inference_s))
            action = np.asarray(policy_out.action, dtype=np.float32)
            target_single_shape = tuple(self.env.single_action_space.shape)
            if len(target_single_shape) == 2:
                chunk_len, action_dim = target_single_shape
                if action.ndim == 1:
                    action = action[None, :]
                if action.ndim != 2 or action.shape[-1] != action_dim:
                    raise ValueError(
                        "Policy action shape is incompatible with chunked Allegro action space: "
                        f"got {action.shape}, expected batch x {action_dim}"
                    )
                if action.shape[0] != n_envs:
                    if action.shape[0] == 1 and n_envs > 1:
                        action = np.repeat(action, repeats=n_envs, axis=0)
                    else:
                        raise ValueError(
                            f"Expected action batch {n_envs}, got shape {action.shape}"
                        )

            else:
                target_action_ndim = len(target_single_shape)
                if action.ndim == target_action_ndim:
                    action = np.expand_dims(action, axis=0)
                if action.shape[0] != n_envs:
                    if action.shape[0] == 1 and n_envs > 1:
                        action = np.repeat(action, repeats=n_envs, axis=0)
                    else:
                        raise ValueError(
                            f"Expected action batch {n_envs}, got shape {action.shape}"
                        )
            action = self._policy_action_to_env_target(action, q_a_ref=q_a_ref)
            if self.cfg.action_mode != "absolute":
                q_a_ref = self._update_q_a_reference_from_target(q_a_ref, action)
            if len(target_single_shape) == 2:
                # Like `iiwa_runner`, convert the policy's delta command into one
                # absolute joint target, then repeat it across the env chunk.
                action = np.repeat(action[:, None, :], repeats=chunk_len, axis=1)

            for _ in range(max(1, int(self.cfg.n_repeat))):
                obs, reward, terminated, truncated, info = self.env.step(action)
            env_timing_summary = self._latest_env_timing_summary()
            for key in timing_history:
                if key == "policy_inference_s":
                    continue
                timing_history[key].append(float(env_timing_summary.get(key, np.nan)))

            reward = np.asarray(reward, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            done = np.logical_or(terminated, truncated)
            last_reward = reward.copy()
            last_terminated = terminated.copy()
            last_truncated = truncated.copy()
            last_info = info
            last_step_index = step_index

            episode_rewards[~done_flags] += reward[~done_flags]
            max_rewards = np.maximum(max_rewards, reward)
            done_flags = np.logical_or(done_flags, done)

            videos["obs"].append(self._policy_rgb_from_obs(obs))
            if policy_out.info is not None and "policy_vis" in policy_out.info:
                videos["policy"].append(
                    self._policy_vis_frame(policy_out.info["policy_vis"])
                )

            traj["obs"].append({k: np.asarray(v) for k, v in obs.items()})
            traj["actions"].append(action.copy())
            traj["q_robot"].append(self._current_q_a_from_obs(obs).copy())
            traj["q_reference"].append(q_a_ref.copy())
            traj["rewards"].append(reward.copy())
            traj["terminated"].append(terminated.copy())
            traj["truncated"].append(truncated.copy())
            # Per-step raw artifacts when MotionPolicyCfg.save_artifacts is on.
            _raw = (policy_out.info or {}).get("raw_artifacts") if policy_out.info else None
            if _raw is not None:
                for _k, _v in _raw.items():
                    traj.setdefault(f"raw__{_k}", []).append(np.asarray(_v))
                    traj.setdefault(f"raw__{_k}__step", []).append(np.int32(step_index))

            feedback = policy.observe_rollout_feedback(
                self._policy_state_observation(obs, step_index=step_index + 1)
            )
            del feedback

            if rr is not None and self.server_uri is not None:
                rr.set_time("frame_index", timestamp=step_index)
                rgb = self._policy_rgb_from_obs(obs)
                rr.log(
                    "vis/obs",
                    rr.Image(self._to_uint8(rgb[0] if rgb.ndim == 4 else rgb)),
                )
                if policy_out.info is not None and "policy_vis" in policy_out.info:
                    rr.log(
                        "vis/policy",
                        rr.Image(
                            self._to_uint8(
                                self._policy_vis_frame(policy_out.info["policy_vis"])
                            )
                        ),
                    )

            if isinstance(info, dict):
                self._step_debug_write(
                    "[AllegroRunner] "
                    + self._rollout_status_line(
                        step_index=step_index,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )
                )

        run_key = (
            self._sanitize_run_tag(run_tag)
            if run_tag is not None
            else time.strftime("run_%Y%m%d_%H%M%S")
        )
        save_dir = Path(self.cfg.output_dir).expanduser() / run_key
        if self.cfg.save_videos:
            for key, frames in videos.items():
                if not frames:
                    continue
                self._save_video(
                    frames,
                    save_dir / "videos" / f"{key}.mp4",
                )

        if self.cfg.save_trajectory:
            save_dir.mkdir(parents=True, exist_ok=True)
            def _stack_or_object(values):
                if not values:
                    return np.array([], dtype=object)
                try:
                    return np.stack(values, axis=0)
                except (ValueError, TypeError):
                    out = np.empty(len(values), dtype=object)
                    for i, v in enumerate(values):
                        out[i] = v
                    return out

            extra_raw = {k: _stack_or_object(v) for k, v in traj.items() if k.startswith("raw__")}
            np.savez_compressed(
                save_dir / "trajectory.npz",
                actions=np.asarray(traj["actions"], dtype=np.float32),
                q_robot=np.asarray(traj["q_robot"], dtype=np.float32),
                q_reference=np.asarray(traj["q_reference"], dtype=np.float32),
                rewards=np.asarray(traj["rewards"], dtype=np.float32),
                terminated=np.asarray(traj["terminated"], dtype=bool),
                truncated=np.asarray(traj["truncated"], dtype=bool),
                **extra_raw,
            )
            with open(save_dir / "trajectory_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "render_obs_key": self.render_obs_key,
                        "view_metadata": self._policy_view_metadata(obs),
                        "num_envs": n_envs,
                    },
                    f,
                    indent=2,
                    default=str,
                )

        if self.cfg.save_rrd and rr is not None and self.server_uri is not None:
            if hasattr(rr, "save"):
                rr.save(str(save_dir / "recording.rrd"))

        all_videos = [videos]
        timing_summary = {
            key: float(np.nanmean(values))
            for key, values in timing_history.items()
            if values
        }
        return {
            "demo_returns": episode_rewards,
            "train_returns": episode_rewards,
            "eval_returns": np.array([], dtype=np.float32),
            "videos": videos,
            "all_videos": all_videos,
            "max_rewards": max_rewards,
            "max_reward_mean": float(np.mean(max_rewards)),
            "env_successes": done_flags.copy(),
            "relaxed_successes": done_flags.copy(),
            "last_step_index": last_step_index,
            "last_terminated": last_terminated,
            "last_truncated": last_truncated,
            "last_info": last_info,
            "timing_history": timing_history,
            "timing_summary": timing_summary,
            "save_dir": str(save_dir),
            "demo_keys": [run_key],
        }


def format_run_results(results: dict):
    """Convert raw Allegro rollout results into a display-ready wrapper."""
    return RunResults.from_raw(results)


@dataclass
class RunResults:
    metrics: dict[str, Any]
    videos: dict[str, np.ndarray]
    save_dir: str | None

    @classmethod
    def from_raw(cls, results: dict) -> "RunResults":
        raw_videos = results.get("videos", {})
        videos: dict[str, np.ndarray] = {}
        for key, frames in raw_videos.items():
            stacked = _stack_video_frames(frames)
            if stacked is not None:
                videos[key] = stacked

        metrics = {
            "train_returns": results.get("train_returns"),
            "eval_returns": results.get("eval_returns"),
            "max_rewards": results.get("max_rewards"),
            "max_reward_mean": results.get("max_reward_mean"),
        }
        return cls(
            metrics=metrics,
            videos=videos,
            save_dir=results.get("save_dir"),
        )

    def show(
        self,
        fps: int = 5,
        height: int = 252,
        video_keys: list[str] | None = None,
    ) -> None:
        print("─── Metrics ───")
        for key, value in self.metrics.items():
            if value is not None:
                print(f"  {key}: {value}")
        if self.save_dir:
            print(f"  save_dir: {self.save_dir}")

        keys = video_keys or list(self.videos.keys())
        for key in keys:
            if key not in self.videos:
                continue
            print(f"\n─── {key} ───")
            media.show_videos(self.videos[key], fps=fps, height=height)


def _stack_video_frames(frames: list[np.ndarray]) -> np.ndarray | None:
    if not frames:
        return None
    normalized_frames: list[np.ndarray] = []
    for frame in frames:
        frame_np = np.asarray(frame)
        if frame_np.ndim == 3:
            frame_np = frame_np[None, ...]
        elif frame_np.ndim != 4:
            raise ValueError(
                "Expected video frames with shape (H, W, C) or (N, H, W, C), "
                f"got {frame_np.shape}"
            )
        normalized_frames.append(frame_np)

    target_h = max(frame.shape[1] for frame in normalized_frames)
    target_w = max(frame.shape[2] for frame in normalized_frames)
    n_envs = normalized_frames[0].shape[0]
    n_channels = normalized_frames[0].shape[3]
    dtype = normalized_frames[0].dtype

    out = np.zeros(
        (n_envs, len(frames), target_h, target_w, n_channels),
        dtype=dtype,
    )
    for timestep, frame in enumerate(normalized_frames):
        height, width = frame.shape[1], frame.shape[2]
        if height == target_h and width == target_w:
            out[:, timestep] = frame
        else:
            out[:, timestep, :height, :width] = frame
    return out
