"""Evaluate a mimicgen task THROUGH the client-server protocol, reusing the live server.

The MimicgenRunner owns the full eval harness (reset-to-demo, demo-state warmup, success tracking,
videos, HTML-able results). We give it a RemotePolicy whose predict_action forwards inference to
the running policy server — so the 16B omni model is evaluated on ONE GPU (the server's), no reload,
no in-process FSDP. The runner calls the policy per env step; RemotePolicy maintains a local context
window + action queue and calls the server's chunked `infer` only on refill.

    MUJOCO_GL=egl python -m vera.controller.run_mimicgen_eval \
        --host 127.0.0.1 --port 8800 \
        --dataset /path/to/data/kitti/mimicgen/core/stack_three_d0.hdf5 \
        --num-demos 10 --rollout-horizon 200
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from vera.policy.base_policy import BasePolicy, PolicyObservation, PolicyOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("vera.mimicgen_eval")


class RemotePolicy(BasePolicy):
    """A BasePolicy whose inference runs on the remote server. Bridges the runner's per-step
    predict_action to the server's chunked infer: accumulate a context window of the runner's
    (already view-concatenated) rgb, call infer when the action queue is empty, pop one per step."""

    cfg = None
    device = None

    def __init__(self, client, *, view_keys: List[str], view_widths: List[int], context_frames: int,
                 prompt: Optional[str] = None, verbose: bool = True):
        self._client = client
        self._view_keys = list(view_keys)
        self._view_widths = list(view_widths)
        self._window: deque = deque(maxlen=int(context_frames))
        self._queue: deque = deque()
        self._session = str(uuid.uuid4())
        self.prompt = prompt   # per-rollout WAN prompt; set/change anytime (no server restart needed)
        # Per-chunk progress feedback (the dream/denoise runs server-side, so the notebook is
        # otherwise silent between chunks). Tail the server log for the live denoise %.
        self._verbose = bool(verbose)
        self._chunk = 0
        self._step = 0

    @staticmethod
    def _to_uint8(rgb) -> np.ndarray:
        a = np.asarray(rgb)
        if a.ndim == 4:
            a = a[0]                       # (1,H,W,3) -> (H,W,3)
        if np.issubdtype(a.dtype, np.floating):
            a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
        return np.ascontiguousarray(a.astype(np.uint8))

    def reset(self) -> None:
        self._window.clear()
        self._queue.clear()
        self._session = str(uuid.uuid4())
        self._chunk = 0
        self._step = 0
        self._client.reset({"session_id": self._session, "reason": "eval_episode"})

    def warmup_obs(self, obs: PolicyObservation) -> None:
        self._window.append(self._to_uint8(obs.rgb))      # fill context, no inference

    def predict_action(self, obs: PolicyObservation) -> PolicyOutput:
        self._step += 1
        self._window.append(self._to_uint8(obs.rgb))
        if not self._queue:
            context_rgb = np.stack(list(self._window), axis=0)     # (T,H,W,3) uint8
            req = {
                "context_rgb": context_rgb,
                "view_keys": list(obs.view_keys or self._view_keys),
                "view_widths": list(obs.view_widths or self._view_widths),
                "session_id": self._session,
            }
            if self.prompt is not None:
                req["prompt"] = self.prompt
            if self._verbose:
                # Starts the line; the server is now denoising (watch the live denoise % in the
                # server log / the viewer at :vis-port). Completed in-place once the chunk returns.
                print(f"    chunk {self._chunk + 1:>3} (env step {self._step:>3}): dreaming "
                      f"+ denoising on server…", end="", flush=True)
            _t0 = time.time()
            out = self._client.infer(req)
            actions = np.asarray(out["action"], dtype=np.float32)
            for row in actions:
                self._queue.append(row)
            self._chunk += 1
            if self._verbose:
                print(f"\r    chunk {self._chunk:>3} (env step {self._step:>3}): dream done in "
                      f"{time.time() - _t0:4.1f}s → committing {len(actions)} actions "
                      f"(|a|={np.abs(actions).mean():.3f})            ", flush=True)
        action = self._queue.popleft()
        return PolicyOutput(action=action[None, :], info=None)     # (1, D) for the env step

    def get_warm_start_state(self):
        return None

    def set_warm_start_state(self, state) -> None:
        del state


def main():
    ap = argparse.ArgumentParser(description="mimicgen eval via the client-server protocol")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--dataset", required=True, help="robosuite/mimicgen hdf5 (e.g. stack_three_d0.hdf5)")
    ap.add_argument("--num-demos", type=int, default=10, help="number of demo initial states to evaluate")
    ap.add_argument("--rollout-horizon", type=int, default=200, help="max env steps per episode")
    ap.add_argument("--render-size", type=int, default=128)
    ap.add_argument("--context-frames", type=int, default=None,
                    help="override the WAN context length (default: use the server-advertised 1+(N-1)*stride)")
    ap.add_argument("--prompt", default=None,
                    help="per-rollout WAN text conditioning (defaults to the server's launch text)")
    ap.add_argument("--output-dir", default="/path/to/mimicgen_eval")
    ap.add_argument("--save-videos", action="store_true", default=True)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from vera.env_runner.mimicgen_runner import MimicgenRunner, MimicgenRunnerCfg
    from vera.server.protocol.websocket_policy_client import WebsocketClientPolicy

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    meta = client.get_server_metadata()
    view_keys = list(meta["view_keys"])
    logger.info("server: %s + %s | views=%s | git=%s",
                meta.get("planner_model"), meta.get("idm_model"), view_keys, str(meta.get("git_head"))[:8])

    # Use the WAN's real context length (advertised by the server: 1+(N-1)*stride), NOT a hardcoded
    # default. The runner warms up context_frames-1 demo frames to fill the window before the policy
    # acts. Allow an explicit --context-frames override.
    context_frames = int(args.context_frames) if args.context_frames else int(meta.get("context_frames", 9))
    logger.info("context_frames=%d (from server: %s; warmup=%d)",
                context_frames, meta.get("context_frames"), context_frames - 1)

    cfg = MimicgenRunnerCfg(
        env_name="mimicgen", dataset_path=args.dataset, render_size=args.render_size,
        render_obs_key=view_keys, num_demos_to_run=int(args.num_demos),
        max_episode_steps=int(args.rollout_horizon), n_repeat=1, action_scale=1.0,
        save_videos=bool(args.save_videos), save_trajectory=False, save_rrd=False,
        output_dir=args.output_dir, use_stored_model=False,
        demo_warmup_steps=max(context_frames - 1, 0), log_step_debug=False,
    )
    runner = MimicgenRunner(cfg, device="cpu")
    runner.setup_env()

    remote = RemotePolicy(client, view_keys=view_keys,
                          view_widths=[args.render_size] * len(view_keys),
                          context_frames=context_frames, prompt=args.prompt)

    logger.info("running eval: %d demos x horizon %d on %s", args.num_demos, args.rollout_horizon, args.dataset)
    result = runner.run(remote, run_tag="vera_remote_eval")

    succ = np.asarray(result.get("env_successes", []), dtype=bool)
    relaxed = np.asarray(result.get("relaxed_successes", succ), dtype=bool)
    max_r = np.asarray(result.get("max_rewards", []), dtype=float)
    n = len(succ)
    logger.info("=" * 60)
    logger.info("EVAL DONE: %s", os.path.basename(args.dataset))
    logger.info("  success rate:         %d/%d = %.1f%%", int(succ.sum()), n, 100.0 * succ.mean() if n else 0.0)
    if relaxed.size:
        logger.info("  relaxed success rate: %d/%d = %.1f%%", int(relaxed.sum()), n, 100.0 * relaxed.mean())
    if max_r.size:
        logger.info("  max reward mean:      %.3f", float(max_r.mean()))
    logger.info("  videos/results:       %s", result.get("save_dir"))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
