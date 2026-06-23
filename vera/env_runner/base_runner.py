from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from vera.policy.base_policy import BasePolicy

# Optional deps (guarded)
try:
    import rerun as rr
except Exception:
    rr = None


# ------------------------ Config ------------------------ #
@dataclass
class RerunConfig:
    rerun_grpc_port: int = 9985
    rerun_web_port: int = 9856


@dataclass
class BaseRunnerCfg:
    env_name: Literal[""]
    rerun: RerunConfig | None = None


class BaseRunner(ABC):
    env: Any
    device: torch.device

    policy: BasePolicy
    dynamics_cfg: Any
    planner_cfg: Any

    # rerun server uri
    server_uri: str | None

    def __init__(
        self,
        cfg: BaseRunnerCfg,
        device: torch.device = torch.device("cuda:0"),
    ) -> None:
        self.cfg = cfg
        self.device = device

        self.server_uri = self.setup_rerun()
        self.setup_env()

    @abstractmethod
    def setup_env(self):
        raise NotImplementedError()

    @abstractmethod
    def run(self, policy: BasePolicy, options=None, run_tag: str | None = None):
        raise NotImplementedError()

    def setup_rerun(self):
        if not self.cfg.rerun or rr is None:
            return None

        if rr.is_enabled():
            rr.disconnect()
        else:
            rr.init("eval_policy")

        server_uri = rr.serve_grpc(grpc_port=self.cfg.rerun.rerun_grpc_port)
        rr.serve_web_viewer(
            open_browser=False,
            web_port=self.cfg.rerun.rerun_web_port,
            connect_to=server_uri,
        )
        print(f"[Rerun] {server_uri}")

        return server_uri

    def _rerun_enabled(self) -> bool:
        return rr is not None and self.server_uri is not None

    def log_rerun(
        self,
        step: int,
        images: dict[str, Any] | None = None,
        scalars: dict[str, Any] | None = None,
    ) -> None:
        if not self._rerun_enabled():
            return

        rr.set_time("frame_index", timestamp=step)

        if images:
            for name, img in images.items():
                if img is None:
                    continue
                rr.log(name, rr.Image(img))

        if scalars:
            for name, value in scalars.items():
                if value is None:
                    continue
                rr.log(name, rr.Scalars(value))

    @staticmethod
    def _to_jsonable(obj: Any):
        if is_dataclass(obj):
            return {k: BaseRunner._to_jsonable(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: BaseRunner._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BaseRunner._to_jsonable(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
