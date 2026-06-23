import tempfile
from pathlib import Path
from typing import Any, Tuple, cast

import torch
from omegaconf import DictConfig, OmegaConf

from vera.datasets.normalization import resolve_action_normalization_mode
from vera.utils.ckpt_utils import _DOWNLOADED_ROOT, download_checkpoint

from .motion_policy_types import ModelCheckpoint, PlannerCfg


def load_checkpoint(
    cfg: ModelCheckpoint,
    device: torch.device,
) -> Tuple[Path, dict[str, Any]]:
    del device
    run_path = f"{cfg.entity}/{cfg.project}/{cfg.run_id}"
    ckpt_path, model_cfg = cast(
        Tuple[Path, dict[str, Any]],
        download_checkpoint(
            run_path,
            _DOWNLOADED_ROOT,
            option=cfg.option,
            return_config=True,
            force_redownload=cfg.force_redownload,
        ),
    )
    return ckpt_path, model_cfg


def _extract_normalization_metadata(
    config_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(config_dict, dict):
        return {}

    dataset_cfg = config_dict.get("dataset")
    if isinstance(dataset_cfg, DictConfig):
        dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    if not isinstance(dataset_cfg, dict):
        return {}

    meta = {
        "oflow_scale": dataset_cfg.get("oflow_scale"),
        "oflow_std": dataset_cfg.get("oflow_std"),
        "flow_normalization_mode": dataset_cfg.get("flow_normalization_mode"),
        "oflow_percentile": dataset_cfg.get("oflow_percentile"),
        "oflow_percentile_low": dataset_cfg.get("oflow_percentile_low"),
        "oflow_percentile_high": dataset_cfg.get("oflow_percentile_high"),
        "oflow_percentile_min": dataset_cfg.get("oflow_percentile_min"),
        "oflow_percentile_max": dataset_cfg.get("oflow_percentile_max"),
        "oflow_abs_scale": dataset_cfg.get("oflow_abs_scale"),
        "flow_normalization_space": dataset_cfg.get(
            "flow_normalization_space",
            "raw_fullres",
        ),
        "du_scale": dataset_cfg.get("du_scale", 1.0),
        "action_pre_scale": dataset_cfg.get(
            "action_pre_scale",
            dataset_cfg.get("du_scale", 1.0),
        ),
        "action_mean": dataset_cfg.get("action_mean"),
        "action_std": dataset_cfg.get("action_std"),
        "action_min": dataset_cfg.get("action_min"),
        "action_max": dataset_cfg.get("action_max"),
        "action_percentile": dataset_cfg.get("action_percentile"),
        "action_abs_scale": dataset_cfg.get("action_abs_scale"),
        "action_mode": dataset_cfg.get("action_mode"),
        "robot_name": dataset_cfg.get("robot_name"),
        # Pre-computed state bounds for datasets that normalize positions to
        # [-1, 1] via state_q_min/state_q_max (e.g. pusht_packed). Needed at
        # inference to convert normalized-position actions back to pixel/joint
        # units when the dataset itself doesn't apply action_abs_scale.
        "state_q_min": dataset_cfg.get("state_q_min"),
        "state_q_max": dataset_cfg.get("state_q_max"),
    }
    meta["action_normalization_mode"] = dataset_cfg.get(
        "action_normalization_mode",
        resolve_action_normalization_mode(
            meta["action_mean"],
            meta["action_std"],
            meta["action_min"],
            meta["action_max"],
            meta["action_abs_scale"],
        ),
    )
    return {k: v for k, v in meta.items() if v is not None}


def _summarize_normalization_metadata(meta: dict[str, Any]) -> str:
    if not meta:
        return "none"
    ordered_keys = [
        "flow_normalization_space",
        "flow_normalization_mode",
        "oflow_scale",
        "oflow_std",
        "oflow_percentile",
        "oflow_percentile_low",
        "oflow_percentile_high",
        "oflow_percentile_min",
        "oflow_percentile_max",
        "oflow_abs_scale",
        "action_normalization_mode",
        "action_pre_scale",
        "action_mean",
        "action_std",
        "action_min",
        "action_max",
        "action_percentile",
        "action_abs_scale",
    ]
    parts = [f"{key}={meta[key]}" for key in ordered_keys if key in meta]
    return ", ".join(parts) if parts else "none"


def _default_wan_config_path() -> Path | None:
    self_file = Path(__file__).resolve()
    for base in [self_file.parents[3], self_file.parents[2], Path.cwd()]:
        candidate = (
            base
            / "third_party"
            / "flow-planner"
            / "configurations"
            / "algorithm"
            / "wan_t2v.yaml"
        )
        if candidate.exists():
            return candidate
    cwd_candidate = Path.cwd() / "configurations" / "algorithm" / "wan_t2v.yaml"
    if cwd_candidate.exists():
        return cwd_candidate
    return None


def _load_algorithm_config_from_path(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Algorithm config not found: {path}")
    loaded = OmegaConf.load(path)

    # Two supported yaml shapes:
    # 1. *Wrapped* — a full Lightning config dump (e.g. wandb ``run.config``
    #    serialized to yaml) with top-level ``algorithm:`` / ``dataset:`` /
    #    ``experiment:`` sections. We preserve all sections so dataset
    #    normalization metadata flows through to the runtime
    #    ``_extract_normalization_metadata`` reader.
    # 2. *Unwrapped* — a bare algorithm-architecture yaml (the historical
    #    WAN flow). We wrap it under ``algorithm:`` and stub out the
    #    placeholder dataset/experiment sections.
    is_wrapped = (
        isinstance(loaded, (DictConfig, dict))
        and "algorithm" in cast(Any, loaded)
    )

    if is_wrapped:
        resolved = OmegaConf.to_container(loaded, resolve=True)
        if not isinstance(resolved, dict):
            resolved = {}
        algo = resolved.get("algorithm")
        if not isinstance(algo, dict):
            algo = {}
        if algo.get("name") is None:
            algo["name"] = "wan_t2v"
        if path.stem == "wan_i2v":
            algo["name"] = "wan_i2v"
        algo.setdefault("debug", False)
        resolved["algorithm"] = algo
        return resolved

    if path.stem == "wan_i2v":
        base_path = path.parent / "wan_t2v.yaml"
        if base_path.exists():
            base_loaded = OmegaConf.load(base_path)
            loaded = OmegaConf.merge(base_loaded, loaded)
    stub = OmegaConf.create(
        {
            "experiment": {"training": {"lr": 1e-4}},
            "dataset": {
                "load_video_latent": False,
                "load_prompt_embed": False,
                "n_frames": 17,
                "height": 256,
                "width": 256,
                "fps": 10,
            },
        }
    )
    merged = OmegaConf.merge(stub, {"algorithm": loaded})
    resolved = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(resolved, dict):
        resolved = {}
    algo = resolved.get("algorithm", resolved)
    if not isinstance(algo, dict):
        algo = {}
    if algo.get("name") is None:
        algo["name"] = "wan_t2v"
    if path.stem == "wan_i2v":
        algo["name"] = "wan_i2v"
    algo.setdefault("debug", False)
    return {"algorithm": algo}


def _infer_wan_model_config_from_checkpoint(ckpt_path: Path):
    try:
        ckpt = torch.load(
            str(ckpt_path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception:
        return None
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else None
    if not isinstance(state_dict, dict):
        return None

    def find_shape(*substrings):
        for key, value in state_dict.items():
            if all(substr in key for substr in substrings) and hasattr(value, "shape"):
                return value.shape
        return None

    patch_shape = find_shape("patch_embedding", "weight")
    if patch_shape is None or len(patch_shape) < 2:
        return None
    dim = int(patch_shape[0])
    in_dim = int(patch_shape[1])
    block_indices = set()
    for key in state_dict:
        if ".blocks." in key:
            parts = key.split(".blocks.")[-1].split(".")
            if parts and parts[0].isdigit():
                block_indices.add(int(parts[0]))
    num_layers = max(block_indices) + 1 if block_indices else None
    ffn_shape = find_shape("blocks", "ffn", "0.weight")
    ffn_dim = int(ffn_shape[0]) if ffn_shape is not None and len(ffn_shape) >= 1 else None
    num_heads = dim // 128 if dim % 128 == 0 else 12
    result = {"dim": dim, "in_dim": in_dim, "num_heads": num_heads}
    if num_layers is not None:
        result["num_layers"] = num_layers
    if ffn_dim is not None:
        result["ffn_dim"] = ffn_dim
    return result


def _resolve_flow_planner_paths(cfg: DictConfig, root: Path) -> None:
    path_keys = [
        "algorithm.text_encoder.ckpt_path",
        "algorithm.vae.ckpt_path",
        "algorithm.model.ckpt_path",
        "algorithm.model.tuned_ckpt_path",
        "algorithm.flow_decoder.ckpt_path",
        "algorithm.clip.ckpt_path",
    ]
    for key in path_keys:
        try:
            value = OmegaConf.select(cfg, key)
            if isinstance(value, str) and value and not Path(value).is_absolute():
                OmegaConf.update(cfg, key, str((root / value).resolve()), merge=True)
        except Exception:
            pass


def _is_wan_motion_planner(name: str | None) -> bool:
    return name in ("wan_t2v", "wan_i2v", "wan_ar_df", "wan_ar_tf")


def _apply_wan_planner_overrides(
    cfg: DictConfig,
    planner_cfg: PlannerCfg,
    *,
    sample_steps: int,
) -> None:
    if planner_cfg.flow_planner_data_root:
        _resolve_flow_planner_paths(cfg, Path(planner_cfg.flow_planner_data_root))

    OmegaConf.update(cfg, "algorithm.sample_steps", sample_steps, merge=True)
    OmegaConf.update(cfg, "algorithm.tracker.backend", planner_cfg.tracker_backend, merge=True)
    OmegaConf.update(cfg, "algorithm.tracker.enabled", planner_cfg.tracker_enabled, merge=True)
    OmegaConf.update(
        cfg,
        "algorithm.tracker.return_visualization",
        planner_cfg.tracker_return_visualization,
        merge=True,
    )
    OmegaConf.update(cfg, "algorithm.alltracker.enabled", planner_cfg.alltracker_enabled, merge=True)
    OmegaConf.update(
        cfg,
        "algorithm.alltracker.return_visualization",
        planner_cfg.alltracker_return_visualization,
        merge=True,
    )
    OmegaConf.update(cfg, "algorithm.alltracker.chunk_size", planner_cfg.alltracker_chunk_size, merge=True)
    OmegaConf.update(cfg, "algorithm.alltracker.rate", planner_cfg.alltracker_rate, merge=True)
    OmegaConf.update(
        cfg,
        "algorithm.alltracker.query_frame",
        planner_cfg.alltracker_query_frame,
        merge=True,
    )
    OmegaConf.update(
        cfg,
        "algorithm.alltracker.inference_iters",
        planner_cfg.alltracker_inference_iters,
        merge=True,
    )
    OmegaConf.update(cfg, "algorithm.alltracker.conf_thr", planner_cfg.alltracker_conf_thr, merge=True)
    OmegaConf.update(
        cfg,
        "algorithm.alltracker.bkg_opacity",
        planner_cfg.alltracker_bkg_opacity,
        merge=True,
    )
    OmegaConf.update(
        cfg,
        "algorithm.alltracker.temporal_stride",
        planner_cfg.alltracker_temporal_stride,
        merge=True,
    )
    OmegaConf.update(cfg, "algorithm.cotracker.model_name", planner_cfg.cotracker_model_name, merge=True)
    OmegaConf.update(cfg, "algorithm.cotracker.grid_size", planner_cfg.cotracker_grid_size, merge=True)

    if planner_cfg.flow_decoder_ckpt is not None:
        flow_ckpt_path, _ = load_checkpoint(planner_cfg.flow_decoder_ckpt, torch.device("cpu"))
        OmegaConf.update(cfg, "algorithm.flow_decoder.ckpt_path", str(flow_ckpt_path), merge=True)
        OmegaConf.update(cfg, "algorithm.flow_decoder.enabled", True, merge=True)

    if planner_cfg.flow_decoder_ckpt_path is not None:
        path = Path(planner_cfg.flow_decoder_ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Flow decoder checkpoint not found: {path}")
        OmegaConf.update(cfg, "algorithm.flow_decoder.ckpt_path", str(path.resolve()), merge=True)
        OmegaConf.update(cfg, "algorithm.flow_decoder.enabled", True, merge=True)


def _write_resolved_algorithm_config(algo_cfg: Any) -> Path:
    resolved_algo_cfg = OmegaConf.create(OmegaConf.to_container(algo_cfg, resolve=True))
    handle = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
    handle.close()
    path = Path(handle.name)
    OmegaConf.save(resolved_algo_cfg, path)
    return path


def _resolve_planner_checkpoint_and_config(
    planner_cfg: PlannerCfg,
    device: torch.device,
    *,
    verbose: bool = True,
) -> Tuple[Path | None, dict, str]:
    ckpt = planner_cfg.ckpt
    ckpt_path = planner_cfg.ckpt_path
    algo_config_path = planner_cfg.algorithm_config_path

    if ckpt is not None and ckpt_path is not None:
        raise ValueError(
            "PlannerCfg: set either ckpt (wandb) or ckpt_path (local), not both."
        )

    if ckpt is not None:
        path, downloaded_config_dict = load_checkpoint(ckpt, device)
        config_dict = (
            _load_algorithm_config_from_path(algo_config_path)
            if algo_config_path
            else downloaded_config_dict
        )
        return Path(path), config_dict, "wandb"

    if ckpt_path is not None:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Planner checkpoint not found: {path}")
        if algo_config_path:
            config_dict = _load_algorithm_config_from_path(algo_config_path)
        else:
            sidecar = path.parent / "config.yaml"
            if not sidecar.exists():
                sidecar = path.parent / "config.yml"
            if not sidecar.exists():
                sidecar = path.parent / "run_config.yaml"
            if sidecar.exists():
                config_dict = _load_algorithm_config_from_path(sidecar)
            else:
                default_path = _default_wan_config_path()
                config_dict = (
                    _load_algorithm_config_from_path(default_path)
                    if default_path
                    else {
                        "algorithm": {
                            "name": "wan_t2v",
                            "debug": False,
                            "height": 256,
                            "width": 256,
                            "diffusion_forcing": {"N": 4, "M": 4},
                        }
                    }
                )
        inferred = _infer_wan_model_config_from_checkpoint(path)
        if inferred and isinstance(config_dict.get("algorithm"), dict):
            algo = config_dict["algorithm"]
            model_cfg = algo.get("model")
            if not isinstance(model_cfg, dict):
                model_cfg = (
                    OmegaConf.to_container(model_cfg, resolve=True)
                    if model_cfg is not None
                    else {}
                )
            if not isinstance(model_cfg, dict):
                model_cfg = {}
            algo["model"] = {**model_cfg, **inferred}
            if verbose:
                print(
                    "Inferred WAN model from checkpoint: "
                    f"dim={inferred['dim']}, in_dim={inferred['in_dim']}, "
                    f"num_layers={inferred.get('num_layers')}, "
                    f"ffn_dim={inferred.get('ffn_dim')}"
                )
        return path, config_dict, "local"

    if algo_config_path:
        config_dict = _load_algorithm_config_from_path(algo_config_path)
    else:
        default_path = _default_wan_config_path()
        if default_path:
            config_dict = _load_algorithm_config_from_path(default_path)
        else:
            config_dict = {
                "algorithm": {
                    "name": "wan_t2v",
                    "debug": False,
                    "height": 256,
                    "width": 256,
                    "diffusion_forcing": {"N": 4, "M": 4},
                }
            }
    return None, config_dict, "raw"
