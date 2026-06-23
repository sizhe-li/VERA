"""
MimicGen HDF5 playback utilities: load states, build initial_state for reset_to,
optionally override textures in the stored model XML, and replay episodes.

Use with use_stored_model=True for correct temporal alignment. Optionally pass
texture_overrides to change table (or other) texture while keeping scene/poses.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from robomimic.utils import obs_utils as ObsUtils


def _resolve_texture_path(path: str) -> str:
    """Resolve relative texture paths (e.g. ../textures/dark-wood.png) to absolute using robosuite assets."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    path = path.replace("\\", "/")
    try:
        import robosuite

        # XML paths like "../textures/foo" are relative to arena dir (models/assets/arenas/)
        base = (
            Path(robosuite.__file__).resolve().parent / "models" / "assets" / "arenas"
        )
        resolved = (base / path).resolve()
        if resolved.exists():
            return str(resolved)
    except Exception:
        pass
    return path


def load_states_from_demo(demo_group, demo_key: Optional[str] = None) -> np.ndarray:
    """Load states array from HDF5. Handles flat 'states' dataset and MimicGen dict-style 'states/<key>' group.

    Args:
        demo_group: HDF5 group for one episode (e.g. f["data"][demo_key]), or an open h5py.File when demo_key is set.
        demo_key: If set, demo_group must be the h5py.File; states are loaded by path "data/<demo_key>/states"
                  first (avoids 'Unable to synchronously open object' in some HDF5 files).
    """
    h5file = None
    if demo_key is not None:
        h5file = demo_group
        demo_group = h5file["data"][demo_key]
        # Try path-based access first (can avoid broken group references)
        path_prefix = f"data/{demo_key}"
        try:
            node = h5file.get(f"{path_prefix}/states")
            if node is not None:
                arr = _states_node_to_array(node)
                if arr is not None:
                    return arr
        except (KeyError, OSError, TypeError):
            pass
    try:
        grp = demo_group.get("states")
    except Exception as e:
        _raise_states_load_error(demo_group, e)
    if grp is None:
        _raise_states_load_error(demo_group, None)
    try:
        arr = _states_node_to_array(grp)
        if arr is not None:
            return arr
        extra = f" 'states' is type {type(grp).__name__}"
        if isinstance(grp, h5py.Group):
            extra = f" states group keys: {list(grp.keys())}"
        _raise_states_load_error(demo_group, None, extra=extra)
    except (KeyError, OSError) as e:
        _raise_states_load_error(demo_group, e)


def _states_node_to_array(node) -> Optional[np.ndarray]:
    """Convert an HDF5 dataset or group (states/...) to a single state array, or None."""
    if isinstance(node, h5py.Dataset):
        return np.array(node[:])
    if isinstance(node, h5py.Group):
        keys = list(node.keys())
        for k in ("states", "state", "env_states"):
            if k in keys:
                return np.array(node[k][:])
        if len(keys) == 1:
            return np.array(node[keys[0]][:])
    return None


def _raise_states_load_error(demo_group, err, extra: str = ""):
    """Raise a clear error listing available keys when state loading fails."""
    try:
        keys = list(demo_group.keys())
    except Exception:
        keys = ["<could not list>"]
    msg = (
        "Could not load 'states' from demo group. "
        f"Available keys: {keys}. "
        "Use use_stored_model=True and a MimicGen (or compatible) HDF5 with 'states' or 'states/states'. "
        f"{extra}"
    )
    if err is not None:
        raise type(err)(f"{msg} Original error: {err}") from err
    raise KeyError(msg)


def replace_texture_in_model_xml(model_xml: str, texture_overrides: dict) -> str:
    """Replace texture file paths in MuJoCo model XML by texture name.

    Keeps use_stored_model=True (correct scene/poses) while swapping textures
    (e.g. table from ceramic to dark-wood). Paths can be relative to robosuite
    assets or absolute.

    Args:
        model_xml: Full model XML string (e.g. from HDF5 attrs["model_file"]).
        texture_overrides: Dict mapping texture name to new file path, e.g.
            {"tex-ceramic": "../textures/dark-wood.png"}.

    Returns:
        Modified XML string with file="..." replaced for each named texture.
    """
    if not texture_overrides:
        return model_xml
    out = model_xml
    for name, path in texture_overrides.items():
        path = _resolve_texture_path(path)
        path = path.replace("\\", "/")
        # Attribute order can be name then file or file then name
        out = re.sub(
            r'(name="' + re.escape(name) + r'"[^>]*)file="[^"]*"',
            r'\1file="' + path + '"',
            out,
        )
        out = re.sub(
            r'file="[^"]*"([^>]*name="' + re.escape(name) + r'")',
            r'file="' + path + r'"\1',
            out,
        )
    return out


def build_initial_state(
    demo_group,
    use_stored_model: bool = True,
    texture_overrides: Optional[dict] = None,
    h5file: Optional[Any] = None,
    demo_key: Optional[str] = None,
) -> dict:
    """Build initial_state dict for env.reset_to (MimicGen / robomimic playback).

    use_stored_model=True: include states + model + ep_meta from HDF5 (required
    for correct alignment). use_stored_model=False: only states.

    texture_overrides: Optional dict mapping texture name to file path; applied
    to initial_state["model"] when use_stored_model=True so you can change
    table texture (e.g. {"tex-ceramic": "../textures/dark-wood.png"}) while
    keeping the stored scene.

    h5file, demo_key: If both provided, states are loaded by path (avoids
    'Unable to synchronously open object' in some HDF5 files).
    """
    states = (
        load_states_from_demo(h5file, demo_key)
        if (h5file is not None and demo_key is not None)
        else load_states_from_demo(demo_group)
    )
    initial_state = dict(states=states[0])
    if use_stored_model:
        if "model_file" in demo_group.attrs:
            model = demo_group.attrs["model_file"]
            model = model.decode() if isinstance(model, bytes) else model
            if texture_overrides:
                model = replace_texture_in_model_xml(model, texture_overrides)
            initial_state["model"] = model
        if "ep_meta" in demo_group.attrs:
            ep_meta = demo_group.attrs["ep_meta"]
            initial_state["ep_meta"] = (
                ep_meta.decode() if isinstance(ep_meta, bytes) else ep_meta
            )
    return initial_state


def render_one_episode_to_memory(
    wrapper,
    demo_group,
    max_steps: Optional[int] = None,
    use_stored_model: bool = True,
    texture_overrides: Optional[dict] = None,
    action_padding_length: Optional[int] = 0,
    h5file: Optional[Any] = None,
    demo_key: Optional[str] = None,
) -> dict:
    """Replay one episode: reset_to(initial_state) then step(actions); collect rgb + low_dim.

    use_stored_model=True and texture_overrides (e.g. for table texture) keep
    alignment while allowing visual overrides.

    action_padding_length: Number of extra action steps to pad at the end after the demo finishes.
    For padding, the first 6 dims are zero, the 7th dim is repeated from the last original action's 7th dim.
    Only supports 7D action arrays.

    h5file, demo_key: If both provided, states are loaded by path (avoids
    'Unable to synchronously open object' in some HDF5 files).
    """
    states = (
        load_states_from_demo(h5file, demo_key)
        if (h5file is not None and demo_key is not None)
        else load_states_from_demo(demo_group)
    )
    actions = np.array(demo_group["actions"][:])

    if max_steps is not None:
        actions = actions[:max_steps]

    # Pad actions if required
    if action_padding_length is not None and action_padding_length > 0:
        assert (
            actions.shape[1] == 7
        ), "This function only supports 7D actions for padding."
        last_action = actions[-1]
        pad = np.zeros((action_padding_length, 7), dtype=actions.dtype)
        # First 6 dims zero, last is repeated from last_action[6]
        pad[:, 6] = last_action[6]
        actions = np.concatenate([actions, pad], axis=0)

    wrapper.init_state = build_initial_state(
        demo_group,
        use_stored_model=use_stored_model,
        texture_overrides=texture_overrides,
        h5file=h5file,
        demo_key=demo_key,
    )
    obs, _ = wrapper.reset()

    rgb_streams = defaultdict(list)
    lowdim_streams = defaultdict(list)

    def collect(o):
        for k, v in o.items():
            if ObsUtils.key_is_obs_modality(k, "rgb"):
                rgb_streams[k].append(v)
            elif ObsUtils.key_is_obs_modality(k, "low_dim"):
                lowdim_streams[k].append(v)

    collect(obs)
    for act in actions:
        obs, reward, term, trunc, info = wrapper.step(act)
        collect(obs)
        if term or trunc:
            break

    rgb_streams = {k: np.stack(v, axis=0) for k, v in rgb_streams.items()}
    lowdim_streams = {k: np.stack(v, axis=0) for k, v in lowdim_streams.items()}

    return {
        "rgb": rgb_streams,
        "low_dim": lowdim_streams,
        "states": states,
        "actions": actions,
    }


class MimicGenPlaybackWrapper:
    """Thin wrapper for MimicGen playback: reset_to(initial_state) with states / model / ep_meta."""

    def __init__(self, env):
        self.env = env
        self.init_state: Optional[dict] = None

    def reset(self):
        if self.init_state is not None:
            obs = self.env.reset_to(self.init_state)
        else:
            obs = self.env.reset()
            obs = self.env.get_observation() if obs is None else obs
        if obs is None:
            obs = self.env.get_observation()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
