"""Save the vis server's policy_vis video buffer to an mp4 file.

Seeks through each frame in the vis server's buffer, grabs the JPEG
snapshot, decodes it, and writes to an mp4 via OpenCV.

Usage:
    python -m vera.server.save_vis_video \
        --vis-host localhost --vis-port 8779 \
        --output /path/to/repo/outputs/policy_vis.mp4
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vis-host", default="localhost")
    parser.add_argument("--vis-port", type=int, default=8779)
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output mp4 path."
    )
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    base = f"http://{args.vis_host}:{args.vis_port}"

    # Get stats to know how many frames exist.
    with urllib.request.urlopen(f"{base}/stats.json", timeout=5) as resp:
        stats = json.loads(resp.read().decode("utf-8"))
    video_len = stats.get("video_len", 0)
    if video_len == 0:
        print("No frames in vis server buffer.")
        return
    print(f"Vis server has {video_len} frames. Downloading...")

    # Pause playback so cursor doesn't move while we grab frames.
    urllib.request.urlopen(f"{base}/video/pause", timeout=3)

    frames: list[np.ndarray] = []
    for i in range(video_len):
        # Seek to frame i.
        urllib.request.urlopen(f"{base}/video/seek?pos={i}", timeout=3)
        # Small delay to let the MJPEG stream update.
        time.sleep(0.05)
        # Grab the current policy JPEG snapshot by reading from the MJPG
        # stream (first frame only). We use a raw HTTP request to grab
        # just the JPEG bytes.
        req = urllib.request.Request(f"{base}/policy.mjpg")
        with urllib.request.urlopen(req, timeout=5) as resp:
            # MJPEG stream: read until we get a full JPEG frame.
            buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                # JPEG ends with 0xFFD9.
                end = buf.find(b"\xff\xd9")
                if end != -1:
                    # Find JPEG start (0xFFD8).
                    start = buf.find(b"\xff\xd8")
                    if start != -1:
                        jpeg_data = buf[start : end + 2]
                        arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            frames.append(frame)
                    break
        if (i + 1) % 10 == 0 or i == video_len - 1:
            print(f"  {i + 1}/{video_len}")

    if not frames:
        print("Failed to grab any frames.")
        return

    # Resume playback.
    urllib.request.urlopen(f"{base}/video/play", timeout=3)

    # Write mp4.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"Saved {len(frames)} frames to {out_path} ({W}x{H} @ {args.fps}fps)")


if __name__ == "__main__":
    main()
