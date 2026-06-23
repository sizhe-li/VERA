"""Run the vera controller against a policy server.

Hardware-free (replay backend) bring-up:
    python -m vera.controller.run_controller --mode sync \
        --host <server> --port 8000 --backend replay \
        --videos /path/v1.mp4 /path/v2.mp4 /path/v3.mp4 \
        --prompt "pick up the object" --max-steps 20

Real FR3 (on nora): --backend robot (wires droid.robot_env.RobotEnv + your camera reader).
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser(description="vera robot-side controller")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mode", choices=["sync", "async"], default="sync")
    ap.add_argument("--backend", choices=["replay", "robot", "mimicgen-sim"], default="replay")
    ap.add_argument("--videos", nargs="+", default=None, help="replay: one mp4 per view")
    ap.add_argument("--dataset", default=None, help="mimicgen-sim: robosuite hdf5 (env config)")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--action-horizon", type=int, default=10)
    ap.add_argument("--context-frames", type=int, default=9)
    ap.add_argument("--image-h", type=int, default=128)
    ap.add_argument("--per-view-w", type=int, default=192)
    args = ap.parse_args()

    from vera.controller.controller import VeraController
    from vera.server.protocol.websocket_policy_client import WebsocketClientPolicy

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    meta = client.get_server_metadata()
    view_keys = list(meta["view_keys"])

    if args.backend == "replay":
        from vera.controller.robot_iface import ReplayBackend
        assert args.videos, "replay backend needs --videos (one per view)"
        backend = ReplayBackend(args.videos, view_keys)
    elif args.backend == "mimicgen-sim":
        from vera.controller.mimicgen_sim_backend import MimicgenSimBackend
        assert args.dataset, "mimicgen-sim backend needs --dataset (robosuite hdf5)"
        backend = MimicgenSimBackend(view_keys, dataset_path=args.dataset,
                                     context_frames=args.context_frames, max_steps=400)
    else:
        from vera.controller.robot_iface import RobotEnvBackend
        backend = RobotEnvBackend(view_keys)

    ctrl = VeraController(
        client, backend,
        action_horizon=args.action_horizon, context_frames=args.context_frames,
        image_hw=(args.image_h, args.per_view_w),
    )
    run = ctrl.run_sync if args.mode == "sync" else ctrl.run_async
    result = run(prompt=args.prompt, max_steps=args.max_steps)
    logging.info("controller finished: %s", result)


if __name__ == "__main__":
    main()
