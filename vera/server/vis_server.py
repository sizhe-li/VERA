"""Live MJPEG visualization server for the VERA policy server.

Runs a small HTTP server in a daemon thread alongside the websocket policy
server. Exposes:

    /                 — dashboard HTML (auto-refreshing stats + MJPEG feeds)
    /policy.mjpg      — MJPEG stream of the latest policy visualization frame
    /policy_frame.jpg?pos=N — exact policy visualization frame
    /input.mjpg       — MJPEG stream of the observation playback cursor
    /input_frame.jpg?pos=N — exact observation history frame
    /input_live.mjpg  — MJPEG stream of the latest incoming RGB observation
    /flow_diff.mjpg   — desired (model) vs achieved (cv2 DIS) flow side-by-side
    /flow_diff.jpg?pos=N — flow panel for an exact observation frame
    /dream_rollouts.mjpg — recent per-inference context/executed/lookahead strips
    /queue.mjpg       — MJPEG filmstrip of context window thumbnails (policy)
    /queue.mjpg?stream=obs — filmstrip of observation history thumbnails
    /stats.json       — JSON with the latest stats (queue, step, action, etc.)

Playback controls (HTTP GET):
    /video/play, /video/pause, /video/live
    /video/seek?pos=N
    /video/step?delta=±1     — frame-by-frame step
    /video/rate?rate=1.0     — playback rate multiplier (0.25..4)
    /obs/play, /obs/pause, /obs/live, /obs/seek, /obs/step, /obs/rate
    /obs/replay_tail?frames=N — jump to the last N obs frames and play forward
    /sync?on=1               — link policy + obs cursors (keyboard scrubs both)

Design: the policy server pushes frames/stats into a `VisHub` after each
request; the HTTP handlers read from the hub under a lock. Frame encoding
(JPEG) happens at push time so readers are cheap. Achieved optical flow is
computed lazily on-demand from the obs buffer using cv2 DIS — no GPU needed.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BOUNDARY = "okto-vis-boundary"
_JPEG_QUALITY = 80
_STREAM_FPS = 15.0
_HISTORY_LEN = 500
_STRIP_THUMB_HEIGHT = 72
_STRIP_MAX_FRAMES_DEFAULT = 32
_STRIP_SEPARATOR_PX = 2
_POLICY_VIDEO_FPS = 10.0
_POLICY_VIDEO_MAXLEN = 600  # ~60 s of history at 10 fps
_OBS_VIDEO_FPS = 10.0
_OBS_VIDEO_MAXLEN = 240  # enough for the observation context window
_FLOW_DIFF_HEIGHT = 200  # cv2 DIS works at this res for the diff panel
_PLAYBACK_RATES = (0.25, 0.5, 1.0, 2.0, 4.0)
_FLOW_DISPLAY_ABS_CLIP = 1.0e4
_DREAM_ROLLOUT_MAX = 8
_DREAM_THUMB_HEIGHT = 44
_DREAM_FRAME_HEIGHT = 150
_DREAM_BORDER = {
    "context": (245, 80, 80),
    "executed": (80, 150, 255),
    "lookahead": (70, 200, 120),
}


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray | None:
    """Coerce an arbitrary RGB frame into (H, W, 3) uint8."""
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    return arr


def _make_thumbnail(frame: np.ndarray, height: int) -> np.ndarray | None:
    rgb = _to_uint8_rgb(frame)
    if rgb is None:
        return None
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return None
    new_w = max(1, int(round(w * height / h)))
    return cv2.resize(rgb, (new_w, height), interpolation=cv2.INTER_AREA)


def _compose_strip(
    thumbs: list[np.ndarray], separator_px: int = _STRIP_SEPARATOR_PX
) -> np.ndarray | None:
    if not thumbs:
        return None
    height = thumbs[0].shape[0]
    sep = np.full((height, separator_px, 3), 32, dtype=np.uint8)
    panels: list[np.ndarray] = []
    for i, thumb in enumerate(thumbs):
        if thumb.shape[0] != height:
            thumb = cv2.resize(
                thumb,
                (max(1, int(round(thumb.shape[1] * height / thumb.shape[0]))), height),
                interpolation=cv2.INTER_AREA,
            )
        if i > 0:
            panels.append(sep)
        panels.append(thumb)
    return np.concatenate(panels, axis=1)


def _compose_vertical_panels(panels: list[np.ndarray]) -> np.ndarray | None:
    if not panels:
        return None
    width = max(int(panel.shape[1]) for panel in panels)
    rows: list[np.ndarray] = []
    for idx, panel in enumerate(panels):
        if int(panel.shape[1]) < width:
            pad_w = width - int(panel.shape[1])
            pad = np.full((panel.shape[0], pad_w, 3), 18, dtype=np.uint8)
            panel = np.concatenate([panel, pad], axis=1)
        if idx > 0:
            sep = np.full((6, width, 3), 15, dtype=np.uint8)
            rows.append(sep)
        rows.append(panel)
    return np.concatenate(rows, axis=0)


def _encode_jpeg(frame: np.ndarray) -> bytes | None:
    arr = _to_uint8_rgb(frame)
    if arr is None:
        return None
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
    )
    if not ok:
        return None
    return buf.tobytes()


def _encode_jpeg_frames(frames: np.ndarray | None) -> list[bytes]:
    """Encode one RGB frame or a batched [T,H,W,3] RGB sequence."""
    if frames is None:
        return []
    arr = np.asarray(frames)
    if arr.ndim == 4:
        encoded: list[bytes] = []
        for frame in arr:
            jpeg = _encode_jpeg(frame)
            if jpeg is not None:
                encoded.append(jpeg)
        return encoded
    jpeg = _encode_jpeg(arr)
    return [] if jpeg is None else [jpeg]


def _iter_rgb_frames(frames: np.ndarray | None) -> list[np.ndarray]:
    """Return one RGB frame or every frame in a [T,H,W,3] sequence."""
    if frames is None:
        return []
    arr = np.asarray(frames)
    if arr.ndim == 4:
        return [arr[i] for i in range(arr.shape[0])]
    return [arr]


def _normalize_flow_xy(flow: np.ndarray | None) -> np.ndarray | None:
    """Coerce a flow array into (H, W, 2) float32. Accepts (2,H,W) or (H,W,2)."""
    if flow is None:
        return None
    f = np.asarray(flow)
    if f.ndim == 4:
        f = f[0]
    if f.ndim != 3:
        return None
    if f.shape[0] == 2 and f.shape[-1] != 2:
        f = np.transpose(f, (1, 2, 0))
    if f.shape[-1] != 2:
        return None
    return _sanitize_flow_xy(f)


def _sanitize_flow_xy(flow_xy: np.ndarray) -> np.ndarray:
    """Return finite, display-bounded (H, W, 2) float32 flow for OpenCV paths."""
    arr = np.asarray(flow_xy)
    if arr.shape[-1] != 2:
        raise ValueError(f"Expected flow with final dim 2, got {arr.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        f64 = np.asarray(arr, dtype=np.float64)
        f64 = np.nan_to_num(
            f64,
            copy=True,
            nan=0.0,
            posinf=_FLOW_DISPLAY_ABS_CLIP,
            neginf=-_FLOW_DISPLAY_ABS_CLIP,
        )
        np.clip(f64, -_FLOW_DISPLAY_ABS_CLIP, _FLOW_DISPLAY_ABS_CLIP, out=f64)
        return f64.astype(np.float32, copy=False)


def _flow_magnitude(flow_xy: np.ndarray) -> np.ndarray:
    """Overflow-resistant flow magnitude for display/stat paths."""
    f = _sanitize_flow_xy(flow_xy)
    return np.hypot(f[..., 0], f[..., 1])


def _display_max_mag(mag: np.ndarray, fallback: float = 1.0) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        vals = np.asarray(mag, dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        finite = np.clip(finite, 0.0, _FLOW_DISPLAY_ABS_CLIP)
    if finite.size == 0:
        return float(fallback)
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.percentile(finite, 99.0))
    if not np.isfinite(value):
        return float(fallback)
    return max(value, float(fallback))


def _flow_to_color(flow_xy: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """HSV color-coded flow. (H,W,2) → (H,W,3) uint8 RGB."""
    flow_xy = _sanitize_flow_xy(flow_xy)
    fx, fy = flow_xy[..., 0], flow_xy[..., 1]
    mag = np.hypot(fx, fy)
    ang = (np.arctan2(fy, fx) + np.pi) * (180.0 / np.pi) / 2.0  # OpenCV hue [0,180)
    if max_mag is None or not np.isfinite(max_mag) or max_mag <= 0.0:
        max_mag = _display_max_mag(mag)
    val = np.clip(mag / max_mag, 0.0, 1.0) * 255.0
    hsv = np.zeros((*flow_xy.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.nan_to_num(
        ang, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.nan_to_num(
        val, nan=0.0, posinf=255.0, neginf=0.0
    ).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
    """Stamp a small label bar at the top of an RGB image."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (15, 17, 24), -1)
    cv2.putText(
        out, text, (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (231, 233, 238), 1, cv2.LINE_AA,
    )
    return out


def _diff_to_color(diff_mag: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """Single-channel magnitude → RGB heat (uses cv2.COLORMAP_INFERNO)."""
    with np.errstate(over="ignore", invalid="ignore"):
        diff_mag64 = np.asarray(diff_mag, dtype=np.float64)
        diff_mag64 = np.nan_to_num(
            diff_mag64,
            copy=True,
            nan=0.0,
            posinf=_FLOW_DISPLAY_ABS_CLIP,
            neginf=0.0,
        )
        np.clip(diff_mag64, 0.0, _FLOW_DISPLAY_ABS_CLIP, out=diff_mag64)
        diff_mag = diff_mag64.astype(np.float32, copy=False)
    if max_mag is None or not np.isfinite(max_mag) or max_mag <= 0.0:
        max_mag = _display_max_mag(diff_mag)
    norm = np.clip(diff_mag / max_mag, 0.0, 1.0) * 255.0
    bgr = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _safe_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _safe_write_text(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def _json_for_html(data: Any) -> str:
    return json.dumps(data, default=str).replace("<", "\\u003c")


def _static_viewer_html(manifest: dict[str, Any]) -> str:
    template = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>VERA viewer snapshot</title>
  <style>
    :root { color-scheme: dark; --bg:#111217; --panel:#181b22; --line:#2b313d; --fg:#eceff4; --muted:#9aa3b5; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    header { padding:18px 22px 6px; }
    h1 { margin:0 0 5px; font-size:18px; letter-spacing:0; }
    h2 { margin:0 0 10px; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
    main { padding:0 22px 22px; display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:14px; }
    .sub,.meta,.pos,.k { color:var(--muted); }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; margin-top:14px; }
    .card:first-child { margin-top:0; }
    img { width:100%; display:block; background:#090b10; border:1px solid var(--line); border-radius:6px; }
    .player { margin-top:9px; display:grid; grid-template-columns:auto auto 1fr auto auto; gap:8px; align-items:center; }
    button,select { background:#222733; color:var(--fg); border:1px solid #343b4a; border-radius:6px; padding:5px 8px; font:inherit; }
    input[type=range] { width:100%; }
    .stats { display:grid; grid-template-columns:150px minmax(0,1fr); gap:5px 10px; }
    .v { overflow-wrap:anywhere; font-variant-numeric:tabular-nums; }
    .dream-list { display:flex; flex-direction:column; gap:12px; }
    .dream-row { border-top:1px solid var(--line); padding-top:12px; }
    .dream-row:first-child { border-top:0; padding-top:0; }
    .legend { display:flex; flex-wrap:wrap; gap:12px; margin-top:9px; color:var(--muted); }
    .legend span::before { content:""; display:inline-block; width:9px; height:9px; margin-right:5px; border-radius:2px; }
    .context::before { background:#f55050; } .executed::before { background:#5096ff; } .lookahead::before { background:#46c878; }
    @media (max-width:980px) { main { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<script type="application/json" id="snapshot-manifest">__MANIFEST__</script>
<header>
  <h1>VERA viewer snapshot</h1>
  <div class="sub" id="summary"></div>
</header>
<main>
  <section>
    <div class="card"><h2>observation playback</h2><img id="obs-img"><div class="player" data-player="obs"><button data-cmd="play">Pause</button><button data-cmd="start">Start</button><input type="range" min="0" max="0" value="0"><select data-rate><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select><span class="pos">0 / 0</span></div></div>
    <div class="card"><h2>policy visualization</h2><img id="policy-img"><div class="player" data-player="policy"><button data-cmd="play">Pause</button><button data-cmd="start">Start</button><input type="range" min="0" max="0" value="0"><select data-rate><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select><span class="pos">0 / 0</span></div></div>
    <div class="card"><h2>desired vs achieved flow</h2><img id="flow-img"><div class="meta">follows the selected observation frame</div></div>
    <div class="card"><h2>dream rollouts</h2><div class="dream-list" id="dream-list"></div><div class="legend"><span class="context">context frames</span><span class="executed">future frames translated to action</span><span class="lookahead">lookahead frames not executed</span></div></div>
  </section>
  <aside>
    <div class="card"><h2>run metadata</h2><div class="stats" id="metadata"></div></div>
    <div class="card"><h2>viewer assets</h2><div class="stats" id="asset-stats"></div></div>
  </aside>
</main>
<script>
const manifest = JSON.parse(document.getElementById("snapshot-manifest").textContent);
const baseFps = 10;
function fmt(v) { return v === null || v === undefined ? "–" : String(v); }
function rows(el, data) { el.innerHTML = data.map(([k,v]) => `<span class="k">${k}</span><span class="v">${fmt(v)}</span>`).join(""); }
function makePlayer(name, files, imgId, onFrame) {
  const root = document.querySelector(`[data-player="${name}"]`);
  const img = document.getElementById(imgId);
  const state = {files: files || [], pos: 0, playing: true, rate: 1, timer: null};
  const range = root.querySelector("input[type=range]");
  const pos = root.querySelector(".pos");
  const btn = root.querySelector("[data-cmd=play]");
  const rate = root.querySelector("[data-rate]");
  range.max = Math.max(0, state.files.length - 1);
  function show(i) {
    state.pos = Math.max(0, Math.min(Number(i || 0), Math.max(0, state.files.length - 1)));
    if (state.files.length) img.src = state.files[state.pos];
    range.value = state.pos;
    pos.textContent = `${state.pos} / ${Math.max(0, state.files.length - 1)}`;
    if (onFrame) onFrame(state.pos);
  }
  function loop() {
    clearTimeout(state.timer);
    if (!state.playing) return;
    state.timer = setTimeout(() => { show(state.pos >= state.files.length - 1 ? 0 : state.pos + 1); loop(); }, 1000 / (baseFps * state.rate));
  }
  btn.onclick = () => { state.playing = !state.playing; btn.textContent = state.playing ? "Pause" : "Play"; loop(); };
  root.querySelector("[data-cmd=start]").onclick = () => show(0);
  range.oninput = () => { state.playing = false; btn.textContent = "Play"; show(range.value); };
  rate.onchange = () => { state.rate = Number(rate.value || 1); loop(); };
  show(0); loop();
}
function makeDreamPlayer(row, dream) {
  const img = row.querySelector("img"), range = row.querySelector("input"), pos = row.querySelector(".pos"), btn = row.querySelector("[data-cmd=play]");
  let i = 0, playing = true, timer = null;
  range.max = Math.max(0, dream.frames.length - 1);
  function show(next) { i = Math.max(0, Math.min(Number(next || 0), Math.max(0, dream.frames.length - 1))); if (dream.frames.length) img.src = dream.frames[i]; range.value = i; pos.textContent = `${i} / ${Math.max(0, dream.frames.length - 1)}`; }
  function loop() { clearTimeout(timer); if (!playing) return; timer = setTimeout(() => { show(i >= dream.frames.length - 1 ? 0 : i + 1); loop(); }, 500); }
  btn.onclick = () => { playing = !playing; btn.textContent = playing ? "Pause" : "Play"; loop(); };
  row.querySelector("[data-cmd=start]").onclick = () => show(0);
  range.oninput = () => { playing = false; btn.textContent = "Play"; show(range.value); };
  show(0); loop();
}
function renderDreams() {
  const list = document.getElementById("dream-list");
  if (!(manifest.dreams || []).length) { list.innerHTML = '<div class="meta">no dream rollout recorded</div>'; return; }
  manifest.dreams.forEach((dream) => {
    const row = document.createElement("div");
    row.className = "dream-row";
    row.innerHTML = `<div class="meta">dream ${dream.dream_index ?? "?"} | step ${dream.step_index ?? "?"} | context ${dream.context_len} | executed ${dream.exec_horizon} | future ${dream.future_len}</div><div><img></div><div class="player"><button data-cmd="play">Pause</button><button data-cmd="start">Start</button><input type="range" min="0" max="0" value="0"><span class="pos">0 / 0</span></div>`;
    list.appendChild(row);
    makeDreamPlayer(row, dream);
  });
}
document.getElementById("summary").textContent = `${manifest.created_at} | ${manifest.run_dir}`;
rows(document.getElementById("metadata"), Object.entries(manifest.metadata || {}));
rows(document.getElementById("asset-stats"), [["observation frames", (manifest.observation_frames || []).length], ["policy frames", (manifest.policy_frames || []).length], ["flow frames", (manifest.flow_frames || []).length], ["dream rollouts", (manifest.dreams || []).length]]);
makePlayer("obs", manifest.observation_frames, "obs-img", (i) => { const flow = manifest.flow_frames || []; if (flow.length) document.getElementById("flow-img").src = flow[Math.min(i, flow.length - 1)]; });
makePlayer("policy", manifest.policy_frames, "policy-img");
renderDreams();
</script>
</body>
</html>
"""
    return template.replace("__MANIFEST__", _json_for_html(manifest))


def _rollout_frame_to_uint8(frame: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        min_val = float(np.min(finite)) if finite.size else 0.0
        if min_val < -0.05:
            arr = (np.clip(arr, -1.0, 1.0) + 1.0) * 0.5
        else:
            arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _rollout_rgb_frames(value: Any) -> list[np.ndarray]:
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim != 4:
        return []
    if arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] != 3:
        return []
    frames = []
    for idx in range(int(arr.shape[0])):
        frame = _rollout_frame_to_uint8(arr[idx])
        if frame is not None:
            frames.append(frame)
    return frames


def _bordered_thumb(frame: np.ndarray, role: str) -> np.ndarray | None:
    thumb = _make_thumbnail(frame, _DREAM_THUMB_HEIGHT)
    if thumb is None:
        return None
    return _draw_role_border(thumb, role, thickness=3)


def _draw_role_border(frame: np.ndarray, role: str, *, thickness: int = 3) -> np.ndarray:
    out = frame.copy()
    color = _DREAM_BORDER[role]
    cv2.rectangle(
        out,
        (0, 0),
        (max(0, out.shape[1] - 1), max(0, out.shape[0] - 1)),
        color,
        thickness,
    )
    return out


def _text_bar(width: int, text: str, *, height: int = 24) -> np.ndarray:
    width = max(320, int(width))
    bar = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.putText(
        bar,
        text[:260],
        (8, min(height - 6, 17)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (231, 233, 238),
        1,
        cv2.LINE_AA,
    )
    return bar


def _pad_rgb_width(img: np.ndarray, width: int, *, value: int = 12) -> np.ndarray:
    if int(img.shape[1]) >= width:
        return img
    pad_w = width - int(img.shape[1])
    pad = np.full((img.shape[0], pad_w, 3), value, dtype=img.dtype)
    return np.concatenate([img, pad], axis=1)


def _render_dream_rollout_panel(
    dream_rollout: dict[str, Any] | None,
    *,
    step_index: int | None = None,
) -> np.ndarray | None:
    if not isinstance(dream_rollout, dict):
        return None
    context = _rollout_rgb_frames(dream_rollout.get("context_rgb"))
    future = _rollout_rgb_frames(dream_rollout.get("future_rgb"))
    if not context and not future:
        return None
    exec_horizon = max(0, int(dream_rollout.get("exec_horizon", 0) or 0))
    exec_horizon = min(exec_horizon, len(future))

    thumbs: list[np.ndarray] = []
    for frame in context:
        thumb = _bordered_thumb(frame, "context")
        if thumb is not None:
            thumbs.append(thumb)
    for frame in future[:exec_horizon]:
        thumb = _bordered_thumb(frame, "executed")
        if thumb is not None:
            thumbs.append(thumb)
    for frame in future[exec_horizon:]:
        thumb = _bordered_thumb(frame, "lookahead")
        if thumb is not None:
            thumbs.append(thumb)
    strip = _compose_strip(thumbs)
    if strip is None:
        return None

    dream_index = dream_rollout.get("dream_index", "?")
    planner_ctx = dream_rollout.get("planner_context_len", len(context))
    future_len = len(future)
    lookahead = max(0, future_len - exec_horizon)
    sampling_path = dream_rollout.get("planner_sampling_path") or "planner"
    step = "?" if step_index is None else str(step_index)
    header = _text_bar(
        strip.shape[1],
        (
            f"dream {dream_index}  step {step}  "
            f"context {planner_ctx}/{len(context)} red | "
            f"executed {exec_horizon} blue | lookahead {lookahead} green | "
            f"{sampling_path}"
        ),
    )
    return np.concatenate([header, strip], axis=0)


def _resize_rollout_frame(frame: np.ndarray) -> np.ndarray | None:
    arr = _to_uint8_rgb(frame)
    if arr is None:
        return None
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        return None
    target_h = _DREAM_FRAME_HEIGHT
    target_w = max(1, int(round(w * target_h / h)))
    return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _make_dream_rollout_record(
    dream_rollout: dict[str, Any] | None,
    *,
    step_index: int | None = None,
) -> dict[str, Any] | None:
    if not isinstance(dream_rollout, dict):
        return None
    context = [
        frame for frame in (_resize_rollout_frame(f) for f in _rollout_rgb_frames(
            dream_rollout.get("context_rgb")
        ))
        if frame is not None
    ]
    future = [
        frame for frame in (_resize_rollout_frame(f) for f in _rollout_rgb_frames(
            dream_rollout.get("future_rgb")
        ))
        if frame is not None
    ]
    if not context and not future:
        return None
    exec_horizon = max(0, int(dream_rollout.get("exec_horizon", 0) or 0))
    exec_horizon = min(exec_horizon, len(future))
    return {
        "context": context,
        "future": future,
        "exec_horizon": exec_horizon,
        "dream_index": dream_rollout.get("dream_index", "?"),
        "planner_context_len": dream_rollout.get("planner_context_len", len(context)),
        "planner_sampling_path": dream_rollout.get("planner_sampling_path") or "planner",
        "step_index": step_index,
        "total_len": len(context) + len(future),
        "context_adj_mad": dream_rollout.get("context_adj_mad"),
        "future_adj_mad": dream_rollout.get("future_adj_mad"),
    }


def _dream_record_summary(record: dict[str, Any]) -> dict[str, Any]:
    context = record.get("context") or []
    future = record.get("future") or []
    return {
        "record_id": record.get("record_id"),
        "dream_index": record.get("dream_index"),
        "step_index": record.get("step_index"),
        "context_len": len(context),
        "future_len": len(future),
        "exec_horizon": int(record.get("exec_horizon", 0) or 0),
        "total_len": int(record.get("total_len", len(context) + len(future)) or 0),
        "planner_context_len": record.get("planner_context_len"),
        "planner_sampling_path": record.get("planner_sampling_path"),
        "context_adj_mad": record.get("context_adj_mad"),
        "future_adj_mad": record.get("future_adj_mad"),
    }


def _dream_frame_for_pos(
    record: dict[str, Any],
    frame_pos: int,
) -> tuple[np.ndarray, str, int] | None:
    context = record.get("context") or []
    future = record.get("future") or []
    total_len = int(record.get("total_len", len(context) + len(future)) or 0)
    if total_len <= 0:
        return None
    idx = min(max(0, int(frame_pos)), total_len - 1)
    if idx < len(context):
        return context[idx], "context", idx
    future_idx = idx - len(context)
    role = "executed" if future_idx < int(record.get("exec_horizon", 0) or 0) else "lookahead"
    if not future:
        return None
    future_idx = min(max(0, future_idx), len(future) - 1)
    return future[future_idx], role, idx


def _render_dream_timeline_panel(
    record: dict[str, Any],
    *,
    frame_pos: int,
) -> np.ndarray | None:
    picked = _dream_frame_for_pos(record, frame_pos)
    if picked is None:
        return None
    frame, role, idx = picked
    frame = _draw_role_border(frame, role, thickness=5)
    total_len = max(1, int(record.get("total_len", 1) or 1))
    context_len = len(record.get("context") or [])
    exec_horizon = int(record.get("exec_horizon", 0) or 0)
    lookahead = max(0, len(record.get("future") or []) - exec_horizon)
    dream_index = record.get("dream_index", "?")
    step_index = record.get("step_index", "?")
    header = _text_bar(
        frame.shape[1],
        (
            f"dream {dream_index}  step {step_index}  frame {idx}/{total_len - 1}  "
            f"{role}  context {context_len} | executed {exec_horizon} | "
            f"lookahead {lookahead}"
        ),
    )
    frame = _pad_rgb_width(frame, header.shape[1])
    return np.concatenate([header, frame], axis=0)


def _finite_float(value: Any) -> float | None:
    try:
        arr = np.asarray(value)
        if arr.size != 1:
            return None
        scalar = float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def _summarize_numeric_dict(stats: Any) -> dict[str, float] | None:
    if not isinstance(stats, dict):
        return None
    out: dict[str, float] = {}
    for key, value in stats.items():
        scalar = _finite_float(value)
        if scalar is not None:
            out[str(key)] = scalar
    return out or None


def _summarize_array(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        original = np.asarray(value)
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return {
        "shape": [int(dim) for dim in original.shape],
        "norm": float(np.linalg.norm(finite)),
        "max_abs": float(np.max(np.abs(finite))),
        "mean_abs": float(np.mean(np.abs(finite))),
    }


def _summarize_action_debug(action_debug: Any) -> dict[str, Any] | None:
    if not isinstance(action_debug, dict):
        return None
    out: dict[str, Any] = {}
    for key in ("action_pre_clip", "action_pre_gate", "action_final"):
        summary = _summarize_array(action_debug.get(key))
        if summary is not None:
            out[key] = summary
    return out or None


class VisHub:
    """Thread-safe store for the latest server visualization state."""

    def __init__(
        self,
        strip_max_frames: int = _STRIP_MAX_FRAMES_DEFAULT,
        policy_video_fps: float = _POLICY_VIDEO_FPS,
        policy_video_maxlen: int = _POLICY_VIDEO_MAXLEN,
        obs_video_fps: float = _OBS_VIDEO_FPS,
        obs_video_maxlen: int = _OBS_VIDEO_MAXLEN,
    ) -> None:
        self._lock = threading.Lock()
        self._input_jpeg: bytes | None = None
        self._policy_strip_thumbs: deque[np.ndarray] = deque(maxlen=strip_max_frames)
        self._obs_strip_thumbs: deque[np.ndarray] = deque(maxlen=strip_max_frames)
        self._dream_rollouts: deque[dict[str, Any]] = deque(maxlen=_DREAM_ROLLOUT_MAX)
        self._dream_rollout_seq = 0
        self._strip_max_frames = strip_max_frames
        # Rolling video of policy visualization frames. A background thread
        # advances _video_cursor through these at policy_video_fps * rate,
        # holding at the newest frame when caught up.
        self._video_frames: list[bytes] = []
        self._video_cursor: int = 0
        self._video_maxlen = int(policy_video_maxlen)
        self._video_fps = float(policy_video_fps)
        self._video_paused = False
        self._video_rate = 1.0
        self._video_stop = threading.Event()
        # Observation buffer + parallel desired-flow buffers. Index alignment:
        # _obs_frames[i] is the obs frame at predict_request i; _obs_rgb_arrays[i]
        # is the same frame as a downscaled RGB array (uint8) for cv2 DIS;
        # _desired_flow_arrays[i] is the predicted flow surfaced by the policy
        # for that step (None if track-path / unavailable).
        self._obs_frames: list[bytes] = []
        self._obs_rgb_arrays: list[np.ndarray | None] = []
        self._desired_flow_arrays: list[np.ndarray | None] = []
        self._desired_source_rgb_arrays: list[np.ndarray | None] = []
        self._obs_cursor: int = 0
        self._obs_maxlen = int(obs_video_maxlen)
        self._obs_fps = float(obs_video_fps)
        self._obs_paused = False
        self._obs_rate = 1.0
        # Cursor sync — when True, scrubbing one cursor moves the other.
        self._sync_cursors = False
        # cv2 DIS instance (lazy). FAST preset is ~5ms on 320x240.
        self._dis = None
        self._dis_lock = threading.Lock()

        self._video_thread = threading.Thread(
            target=self._video_loop, name="okto-vis-video", daemon=True
        )
        self._video_thread.start()

        self._stats: dict[str, Any] = {
            "started_at": time.time(),
            "total_requests": 0,
            "predict_requests": 0,
            "enqueue_requests": 0,
            "last_request_kind": None,
            "last_step_index": None,
            "last_infer_ms": None,
            "last_action_abs_max": None,
            "action_chunk_remaining": None,
            "queue_len": None,
            "queue_max": None,
            "jacobian_abs_stats": None,
            "track_stats": None,
            "control_track_stats": None,
            "outlier_stats": None,
            "action_debug_summary": None,
            "metadata": {},
        }
        self._infer_ms_history: deque[float] = deque(maxlen=120)
        self._history: deque[dict[str, Any]] = deque(maxlen=_HISTORY_LEN)

    # ------------- writers (called from the policy server) -------------

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        with self._lock:
            self._stats["metadata"] = dict(metadata)

    def record_request(
        self,
        *,
        kind: str,  # "predict" | "enqueue"
        input_rgb: np.ndarray | None = None,
        policy_vis: np.ndarray | None = None,
        step_index: int | None = None,
        infer_ms: float | None = None,
        action: np.ndarray | None = None,
        action_chunk_remaining: int | None = None,
        queue_len: int | None = None,
        queue_max: int | None = None,
        jacobian_abs_stats: dict[str, Any] | None = None,
        track_stats: dict[str, Any] | None = None,
        control_track_stats: dict[str, Any] | None = None,
        outlier_stats: dict[str, Any] | None = None,
        action_debug: dict[str, Any] | None = None,
        dream_rollout: dict[str, Any] | None = None,
        desired_flow: np.ndarray | None = None,
        desired_source_rgb: np.ndarray | None = None,
    ) -> None:
        input_entries: list[tuple[bytes, np.ndarray | None, np.ndarray | None]] = []
        for input_frame in _iter_rgb_frames(input_rgb):
            input_jpeg = _encode_jpeg(input_frame)
            if input_jpeg is None:
                continue
            thumbnail = _make_thumbnail(input_frame, _STRIP_THUMB_HEIGHT)
            obs_small = None
            arr = _to_uint8_rgb(input_frame)
            if arr is not None:
                h, w = arr.shape[:2]
                target_h = _FLOW_DIFF_HEIGHT
                target_w = max(1, int(round(w * target_h / max(h, 1))))
                obs_small = cv2.resize(
                    arr,
                    (target_w, target_h),
                    interpolation=cv2.INTER_AREA,
                )
            input_entries.append((input_jpeg, obs_small, thumbnail))
        policy_jpegs = _encode_jpeg_frames(policy_vis)
        flow_xy = _normalize_flow_xy(desired_flow)
        src_rgb_small = None
        if desired_source_rgb is not None:
            s = _to_uint8_rgb(desired_source_rgb)
            if s is not None:
                src_rgb_small = cv2.resize(
                    s, (
                        input_entries[0][1].shape[1],
                        input_entries[0][1].shape[0],
                    ) if input_entries and input_entries[0][1] is not None
                    else (s.shape[1], s.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
        abs_max = None
        if action is not None:
            try:
                abs_max = float(np.abs(np.asarray(action)).max())
            except (ValueError, TypeError):
                abs_max = None
        track_stats_summary = _summarize_numeric_dict(track_stats)
        control_track_stats_summary = _summarize_numeric_dict(control_track_stats)
        outlier_stats_summary = _summarize_numeric_dict(outlier_stats)
        jacobian_abs_stats_summary = _summarize_numeric_dict(jacobian_abs_stats)
        action_debug_summary = _summarize_action_debug(action_debug)
        dream_rollout_record = _make_dream_rollout_record(
            dream_rollout,
            step_index=step_index,
        )

        entry = {
            "t": time.time(),
            "kind": kind,
            "step_index": step_index,
            "infer_ms": infer_ms,
            "action_abs_max": abs_max,
            "action_chunk_remaining": action_chunk_remaining,
            "queue_len": queue_len,
            "queue_max": queue_max,
            "jacobian_abs_stats": jacobian_abs_stats_summary,
            "track_stats": track_stats_summary,
            "control_track_stats": control_track_stats_summary,
            "outlier_stats": outlier_stats_summary,
            "action_debug_summary": action_debug_summary,
            "dream_rollout": dream_rollout_record is not None,
        }

        with self._lock:
            if input_entries:
                self._input_jpeg = input_entries[-1][0]
                for entry_idx, (input_jpeg, obs_small, thumbnail) in enumerate(
                    input_entries
                ):
                    self._obs_frames.append(input_jpeg)
                    self._obs_rgb_arrays.append(obs_small)
                    self._desired_flow_arrays.append(
                        flow_xy if entry_idx == 0 else None
                    )
                    self._desired_source_rgb_arrays.append(
                        src_rgb_small if entry_idx == 0 else None
                    )
                    if thumbnail is not None:
                        self._obs_strip_thumbs.append(thumbnail)
                overflow = len(self._obs_frames) - self._obs_maxlen
                if overflow > 0:
                    del self._obs_frames[:overflow]
                    del self._obs_rgb_arrays[:overflow]
                    del self._desired_flow_arrays[:overflow]
                    del self._desired_source_rgb_arrays[:overflow]
                    self._obs_cursor = max(0, self._obs_cursor - overflow)
            if policy_jpegs:
                self._video_frames.extend(policy_jpegs)
                for frame in _iter_rgb_frames(policy_vis):
                    thumbnail = _make_thumbnail(frame, _STRIP_THUMB_HEIGHT)
                    if thumbnail is not None:
                        self._policy_strip_thumbs.append(thumbnail)
                overflow = len(self._video_frames) - self._video_maxlen
                if overflow > 0:
                    del self._video_frames[:overflow]
                    self._video_cursor = max(0, self._video_cursor - overflow)
            if dream_rollout_record is not None:
                self._dream_rollout_seq += 1
                dream_rollout_record["record_id"] = self._dream_rollout_seq
                self._dream_rollouts.append(dream_rollout_record)
            s = self._stats
            s["total_requests"] += 1
            if kind == "predict":
                s["predict_requests"] += 1
            elif kind == "enqueue":
                s["enqueue_requests"] += 1
            s["last_request_kind"] = kind
            s["last_step_index"] = step_index
            s["last_infer_ms"] = infer_ms
            s["last_action_abs_max"] = abs_max
            if infer_ms is not None:
                self._infer_ms_history.append(float(infer_ms))
            if action_chunk_remaining is not None:
                s["action_chunk_remaining"] = action_chunk_remaining
            if queue_len is not None:
                s["queue_len"] = queue_len
            if queue_max is not None:
                s["queue_max"] = queue_max
            if jacobian_abs_stats_summary is not None:
                s["jacobian_abs_stats"] = jacobian_abs_stats_summary
            if track_stats_summary is not None:
                s["track_stats"] = track_stats_summary
            if control_track_stats_summary is not None:
                s["control_track_stats"] = control_track_stats_summary
            if outlier_stats_summary is not None:
                s["outlier_stats"] = outlier_stats_summary
            if action_debug_summary is not None:
                s["action_debug_summary"] = action_debug_summary
            self._history.append(entry)

    # ------------- readers (called from the HTTP handlers) -------------

    def snapshot_stats(self) -> dict[str, Any]:
        with self._lock:
            s = dict(self._stats)
            s["uptime_s"] = time.time() - s["started_at"]
            s["history_len"] = len(self._history)
            s["video_len"] = len(self._video_frames)
            # latest dream rollout: count (for the row index) + its timeline length (for animation).
            s["dream_count"] = len(self._dream_rollouts)
            s["dream_len"] = (
                int(self._dream_rollouts[-1].get("total_len", 0) or 0)
                if self._dream_rollouts else 0
            )
            s["video_cursor"] = (
                min(self._video_cursor, max(0, len(self._video_frames) - 1))
                if self._video_frames
                else 0
            )
            s["video_maxlen"] = self._video_maxlen
            s["video_fps"] = self._video_fps
            s["video_paused"] = self._video_paused
            s["video_rate"] = self._video_rate
            s["video_tail_lag"] = (
                max(0, len(self._video_frames) - 1 - s["video_cursor"])
                if self._video_frames
                else 0
            )
            s["obs_len"] = len(self._obs_frames)
            s["obs_cursor"] = (
                min(self._obs_cursor, max(0, len(self._obs_frames) - 1))
                if self._obs_frames
                else 0
            )
            s["obs_maxlen"] = self._obs_maxlen
            s["obs_fps"] = self._obs_fps
            s["obs_paused"] = self._obs_paused
            s["obs_rate"] = self._obs_rate
            s["obs_tail_lag"] = (
                max(0, len(self._obs_frames) - 1 - s["obs_cursor"])
                if self._obs_frames
                else 0
            )
            s["sync_cursors"] = self._sync_cursors
            s["dream_rollout_len"] = len(self._dream_rollouts)
            s["dream_rollout_maxlen"] = _DREAM_ROLLOUT_MAX
            s["dream_frame_max"] = max(
                [int(r.get("total_len", 0) or 0) - 1 for r in self._dream_rollouts]
                or [0]
            )
            s["dream_latest_total_len"] = (
                int(self._dream_rollouts[-1].get("total_len", 0) or 0)
                if self._dream_rollouts
                else 0
            )
            s["dream_rollout_summaries"] = [
                _dream_record_summary(r) for r in self._dream_rollouts
            ]
            s["infer_ms_history"] = list(self._infer_ms_history)
            s["recent_history"] = list(self._history)[-180:]
            now = time.time()
            if self._history:
                s["last_request_age_s"] = max(
                    0.0, now - float(self._history[-1].get("t", now))
                )
                recent_cutoff = now - 5.0
                recent_count = sum(
                    1 for e in self._history if e.get("t", 0) >= recent_cutoff
                )
                s["recent_request_hz"] = recent_count / 5.0
            else:
                s["last_request_age_s"] = None
                s["recent_request_hz"] = 0.0
            s["has_desired_flow"] = any(
                f is not None for f in self._desired_flow_arrays
            )
            return s

    def snapshot_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def snapshot_dream_rollout_summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [_dream_record_summary(r) for r in self._dream_rollouts]

    def snapshot_policy_jpeg(self) -> bytes | None:
        with self._lock:
            if not self._video_frames:
                return None
            idx = min(self._video_cursor, len(self._video_frames) - 1)
            return self._video_frames[idx]

    def snapshot_policy_frame_jpeg(self, pos: int) -> bytes | None:
        with self._lock:
            if not self._video_frames:
                return None
            idx = max(0, min(int(pos), len(self._video_frames) - 1))
            return self._video_frames[idx]

    def snapshot_input_jpeg(self) -> bytes | None:
        with self._lock:
            if self._obs_frames:
                idx = min(self._obs_cursor, len(self._obs_frames) - 1)
                return self._obs_frames[idx]
            return self._input_jpeg

    def snapshot_input_frame_jpeg(self, pos: int) -> bytes | None:
        with self._lock:
            if not self._obs_frames:
                return self._input_jpeg
            idx = max(0, min(int(pos), len(self._obs_frames) - 1))
            return self._obs_frames[idx]

    def snapshot_input_live_jpeg(self) -> bytes | None:
        with self._lock:
            return self._input_jpeg

    def snapshot_strip_jpeg(self, stream: str = "policy") -> bytes | None:
        with self._lock:
            if stream == "obs":
                thumbs = list(self._obs_strip_thumbs)
            else:
                thumbs = list(self._policy_strip_thumbs)
        if not thumbs:
            return None
        strip = _compose_strip(thumbs)
        if strip is None:
            return None
        return _encode_jpeg(strip)

    def snapshot_flow_diff_jpeg(self, pos: int | None = None) -> bytes | None:
        """Render desired vs achieved flow for the current obs cursor."""
        with self._lock:
            n = len(self._obs_rgb_arrays)
            if n == 0:
                return None
            if pos is None:
                idx = min(max(self._obs_cursor, 0), n - 1)
            else:
                idx = min(max(int(pos), 0), n - 1)
            curr = self._obs_rgb_arrays[idx]
            prev = self._obs_rgb_arrays[idx - 1] if idx >= 1 else None
            desired = self._desired_flow_arrays[idx]
            desired_src = self._desired_source_rgb_arrays[idx]
        if curr is None:
            return None
        # Achieved flow via cv2 DIS between prev → curr.
        achieved = None
        if prev is not None:
            try:
                prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
                with self._dis_lock:
                    if self._dis is None:
                        self._dis = cv2.DISOpticalFlow_create(
                            cv2.DISOPTICAL_FLOW_PRESET_FAST
                        )
                    achieved = self._dis.calc(prev_gray, curr_gray, None)
            except Exception as exc:  # pragma: no cover
                logger.debug("DIS flow failed: %s", exc)
                achieved = None
        return _encode_jpeg(_build_flow_diff_panel(curr, desired_src, desired, achieved))

    def snapshot_dream_rollouts_jpeg(
        self,
        frame_pos: int = 0,
        *,
        row_index: int | None = None,
        record_id: int | None = None,
    ) -> bytes | None:
        with self._lock:
            records = list(self._dream_rollouts)
        if record_id is not None:
            records = [r for r in records if r.get("record_id") == record_id]
        elif row_index is not None:
            if row_index < 0 or row_index >= len(records):
                return None
            records = [records[row_index]]
        panels = [
            panel
            for panel in (
                _render_dream_timeline_panel(record, frame_pos=frame_pos)
                for record in records
            )
            if panel is not None
        ]
        canvas = _compose_vertical_panels(panels)
        if canvas is None:
            return None
        return _encode_jpeg(canvas)

    def export_static_snapshot(self, run_dir: str | Path) -> str | None:
        """Write a self-contained offline viewer snapshot into a run folder."""
        run_path = Path(run_dir).expanduser()
        snapshot_dir = run_path / "viewer_snapshot"
        assets_dir = snapshot_dir / "assets"
        try:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            assets_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("viewer snapshot mkdir failed for %s: %s", run_path, exc)
            return None

        stats = self.snapshot_stats()
        with self._lock:
            obs_frames = list(self._obs_frames)
            video_frames = list(self._video_frames)
            obs_rgb_arrays = [
                None if arr is None else np.array(arr, copy=True)
                for arr in self._obs_rgb_arrays
            ]
            desired_flow_arrays = [
                None if arr is None else np.array(arr, copy=True)
                for arr in self._desired_flow_arrays
            ]
            desired_source_rgb_arrays = [
                None if arr is None else np.array(arr, copy=True)
                for arr in self._desired_source_rgb_arrays
            ]
            dream_records = []
            for record in self._dream_rollouts:
                copied = dict(record)
                copied["context"] = [
                    np.array(frame, copy=True) for frame in (record.get("context") or [])
                ]
                copied["future"] = [
                    np.array(frame, copy=True) for frame in (record.get("future") or [])
                ]
                dream_records.append(copied)
            metadata = dict(self._stats.get("metadata", {}) or {})
            history = list(self._history)

        obs_asset_names: list[str] = []
        for idx, jpeg in enumerate(obs_frames):
            name = f"assets/obs_{idx:06d}.jpg"
            _safe_write_bytes(snapshot_dir / name, jpeg or _placeholder_jpeg())
            obs_asset_names.append(name)

        policy_asset_names: list[str] = []
        for idx, jpeg in enumerate(video_frames):
            name = f"assets/policy_{idx:06d}.jpg"
            _safe_write_bytes(snapshot_dir / name, jpeg or _placeholder_jpeg())
            policy_asset_names.append(name)

        flow_asset_names: list[str] = []
        dis = None
        for idx, curr in enumerate(obs_rgb_arrays):
            name = f"assets/flow_{idx:06d}.jpg"
            jpeg = None
            if curr is not None:
                achieved = None
                prev = obs_rgb_arrays[idx - 1] if idx >= 1 else None
                if prev is not None:
                    try:
                        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
                        curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
                        if dis is None:
                            dis = cv2.DISOpticalFlow_create(
                                cv2.DISOPTICAL_FLOW_PRESET_FAST
                            )
                        achieved = dis.calc(prev_gray, curr_gray, None)
                    except Exception as exc:  # pragma: no cover
                        logger.debug("viewer snapshot DIS flow failed: %s", exc)
                        achieved = None
                jpeg = _encode_jpeg(
                    _build_flow_diff_panel(
                        curr,
                        desired_source_rgb_arrays[idx]
                        if idx < len(desired_source_rgb_arrays)
                        else None,
                        desired_flow_arrays[idx]
                        if idx < len(desired_flow_arrays)
                        else None,
                        achieved,
                    )
                )
            _safe_write_bytes(snapshot_dir / name, jpeg or _placeholder_jpeg())
            flow_asset_names.append(name)

        dream_manifests: list[dict[str, Any]] = []
        for row_idx, record in enumerate(dream_records):
            summary = _dream_record_summary(record)
            frames: list[str] = []
            total = max(1, int(summary.get("total_len", 0) or 0))
            record_id = summary.get("record_id")
            stem = f"dream_{row_idx:02d}_{record_id if record_id is not None else 'row'}"
            for pos in range(total):
                name = f"assets/{stem}_{pos:03d}.jpg"
                panel = _render_dream_timeline_panel(record, frame_pos=pos)
                _safe_write_bytes(
                    snapshot_dir / name,
                    (
                        _encode_jpeg(panel)
                        if panel is not None
                        else None
                    )
                    or _placeholder_jpeg(),
                )
                frames.append(name)
            summary["frames"] = frames
            dream_manifests.append(summary)

        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_dir": str(run_path),
            "metadata": metadata,
            "stats": stats,
            "recent_history": history[-180:],
            "observation_frames": obs_asset_names,
            "policy_frames": policy_asset_names,
            "flow_frames": flow_asset_names,
            "dreams": dream_manifests,
        }
        try:
            _safe_write_text(
                assets_dir / "manifest.json",
                json.dumps(manifest, indent=2, default=str) + "\n",
            )
            _safe_write_text(snapshot_dir / "index.html", _static_viewer_html(manifest))
            _safe_write_text(
                run_path / "viewer_snapshot.html",
                '<!doctype html><meta charset="utf-8">'
                '<meta http-equiv="refresh" content="0; url=viewer_snapshot/index.html">'
                '<a href="viewer_snapshot/index.html">viewer_snapshot/index.html</a>\n',
            )
            return str(snapshot_dir / "index.html")
        except OSError as exc:
            logger.warning("viewer snapshot write failed for %s: %s", run_path, exc)
            return None

    def clear_history(self) -> None:
        with self._lock:
            self._policy_strip_thumbs.clear()
            self._obs_strip_thumbs.clear()
            self._dream_rollouts.clear()
            self._dream_rollout_seq = 0
            self._history.clear()
            self._video_frames.clear()
            self._video_cursor = 0
            self._obs_frames.clear()
            self._obs_rgb_arrays.clear()
            self._desired_flow_arrays.clear()
            self._desired_source_rgb_arrays.clear()
            self._obs_cursor = 0
            self._infer_ms_history.clear()

    # ------------- background threads -------------

    def _video_loop(self) -> None:
        # Independent base period — actual advance speed is multiplied by
        # _video_rate / _obs_rate, so the cursor can step >1 frame per tick
        # at higher rates without sleeping shorter.
        period = 1.0 / max(self._video_fps, 0.1)
        accum_v = 0.0
        accum_o = 0.0
        while not self._video_stop.wait(period):
            with self._lock:
                if not self._video_paused and self._video_frames:
                    accum_v += float(self._video_rate)
                    step = int(accum_v)
                    accum_v -= step
                    if step:
                        tail = len(self._video_frames) - 1
                        self._video_cursor = min(self._video_cursor + step, tail)
                else:
                    accum_v = 0.0
                if not self._obs_paused and self._obs_frames:
                    accum_o += float(self._obs_rate)
                    step = int(accum_o)
                    accum_o -= step
                    if step:
                        tail = len(self._obs_frames) - 1
                        self._obs_cursor = min(self._obs_cursor + step, tail)
                else:
                    accum_o = 0.0

    # ------------- policy-vis playback controls -------------

    def video_set_paused(self, paused: bool) -> None:
        with self._lock:
            self._video_paused = bool(paused)
            if self._sync_cursors:
                self._obs_paused = bool(paused)

    def video_seek(self, pos: int) -> None:
        with self._lock:
            if self._video_frames:
                tail = len(self._video_frames) - 1
                self._video_cursor = max(0, min(int(pos), tail))
            if self._sync_cursors and self._obs_frames:
                tail = len(self._obs_frames) - 1
                self._obs_cursor = max(0, min(int(pos), tail))

    def video_step(self, delta: int) -> None:
        with self._lock:
            if self._video_frames:
                tail = len(self._video_frames) - 1
                self._video_cursor = max(0, min(self._video_cursor + int(delta), tail))
                self._video_paused = True
            if self._sync_cursors and self._obs_frames:
                tail = len(self._obs_frames) - 1
                self._obs_cursor = max(0, min(self._obs_cursor + int(delta), tail))
                self._obs_paused = True

    def video_jump_to_live(self) -> None:
        with self._lock:
            if self._video_frames:
                self._video_cursor = len(self._video_frames) - 1
                self._video_paused = False
            if self._sync_cursors and self._obs_frames:
                self._obs_cursor = len(self._obs_frames) - 1
                self._obs_paused = False

    def video_set_rate(self, rate: float) -> None:
        with self._lock:
            self._video_rate = float(max(0.05, min(rate, 8.0)))
            if self._sync_cursors:
                self._obs_rate = self._video_rate

    # ------------- observation playback controls -------------

    def obs_set_paused(self, paused: bool) -> None:
        with self._lock:
            self._obs_paused = bool(paused)
            if self._sync_cursors:
                self._video_paused = bool(paused)

    def obs_seek(self, pos: int) -> None:
        with self._lock:
            if self._obs_frames:
                tail = len(self._obs_frames) - 1
                self._obs_cursor = max(0, min(int(pos), tail))
            if self._sync_cursors and self._video_frames:
                tail = len(self._video_frames) - 1
                self._video_cursor = max(0, min(int(pos), tail))

    def obs_step(self, delta: int) -> None:
        with self._lock:
            if self._obs_frames:
                tail = len(self._obs_frames) - 1
                self._obs_cursor = max(0, min(self._obs_cursor + int(delta), tail))
                self._obs_paused = True
            if self._sync_cursors and self._video_frames:
                tail = len(self._video_frames) - 1
                self._video_cursor = max(0, min(self._video_cursor + int(delta), tail))
                self._video_paused = True

    def obs_jump_to_live(self) -> None:
        with self._lock:
            if self._obs_frames:
                self._obs_cursor = len(self._obs_frames) - 1
                self._obs_paused = False
            if self._sync_cursors and self._video_frames:
                self._video_cursor = len(self._video_frames) - 1
                self._video_paused = False

    def obs_set_rate(self, rate: float) -> None:
        with self._lock:
            self._obs_rate = float(max(0.05, min(rate, 8.0)))
            if self._sync_cursors:
                self._video_rate = self._obs_rate

    def obs_replay_tail(self, frames: int = 60) -> int:
        with self._lock:
            if not self._obs_frames:
                self._obs_cursor = 0
                return 0
            n = max(1, int(frames))
            self._obs_cursor = max(0, len(self._obs_frames) - n)
            self._obs_paused = False
            return self._obs_cursor

    # ------------- sync -------------

    def sync_set(self, on: bool) -> None:
        with self._lock:
            self._sync_cursors = bool(on)


def _build_flow_diff_panel(
    curr_rgb: np.ndarray,
    desired_src_rgb: np.ndarray | None,
    desired_flow: np.ndarray | None,
    achieved_flow: np.ndarray | None,
) -> np.ndarray:
    """3-up panel: desired flow | achieved flow | |Δ| heat. Returns RGB uint8."""
    h, w = curr_rgb.shape[:2]
    blank = np.full((h, w, 3), 30, dtype=np.uint8)

    # Resize desired flow into curr space if shapes differ.
    desired_resized = None
    if desired_flow is not None and desired_flow.size > 0:
        df = _normalize_flow_xy(desired_flow)
        if df is not None:
            if df.shape[:2] != (h, w):
                sx = w / max(df.shape[1], 1)
                sy = h / max(df.shape[0], 1)
                df = cv2.resize(df, (w, h), interpolation=cv2.INTER_LINEAR)
                df[..., 0] *= sx
                df[..., 1] *= sy
            desired_resized = _sanitize_flow_xy(df)

    achieved_resized = None
    if achieved_flow is not None:
        ac = _normalize_flow_xy(achieved_flow)
        if ac is not None:
            if ac.shape[:2] != (h, w):
                sx = w / max(ac.shape[1], 1)
                sy = h / max(ac.shape[0], 1)
                ac = cv2.resize(ac, (w, h), interpolation=cv2.INTER_LINEAR)
                ac[..., 0] *= sx
                ac[..., 1] *= sy
            achieved_resized = _sanitize_flow_xy(ac)

    # Pick a shared max magnitude so all three panels share a scale.
    mags = []
    for fxy in (desired_resized, achieved_resized):
        if fxy is not None:
            mags.append(_display_max_mag(_flow_magnitude(fxy)))
    shared_max = max(mags) if mags else 1.0
    shared_max = max(shared_max, 1.0) if np.isfinite(shared_max) else 1.0

    desired_panel = (
        _flow_to_color(desired_resized, max_mag=shared_max)
        if desired_resized is not None else blank
    )
    achieved_panel = (
        _flow_to_color(achieved_resized, max_mag=shared_max)
        if achieved_resized is not None else blank
    )
    if desired_resized is not None and achieved_resized is not None:
        diff = _flow_magnitude(desired_resized - achieved_resized)
        diff_panel = _diff_to_color(diff, max_mag=shared_max)
    else:
        diff_panel = blank.copy()

    desired_panel = _label_panel(
        desired_panel,
        "DESIRED (model)" if desired_resized is not None else "desired flow unavailable",
    )
    achieved_panel = _label_panel(
        achieved_panel,
        "ACHIEVED (cv2 DIS)" if achieved_resized is not None else "need 2 obs frames",
    )
    diff_panel = _label_panel(
        diff_panel,
        f"|Δ| max={shared_max:.1f}px" if (
            desired_resized is not None and achieved_resized is not None
        ) else "diff: need both flows",
    )
    sep = np.full((h, _STRIP_SEPARATOR_PX, 3), 32, dtype=np.uint8)
    return np.concatenate(
        [desired_panel, sep, achieved_panel, sep, diff_panel], axis=1
    )


# ──────────────────────────────────────────────────────────────────────
# HTTP server
# ──────────────────────────────────────────────────────────────────────

_DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>vera viewer</title>
<style>
 html,body{margin:0;background:#0d0d0f;color:#d4d4d8;font-family:ui-monospace,Menlo,monospace;font-size:13px}
 #params{padding:9px 14px;border-bottom:1px solid #222;letter-spacing:.2px;white-space:nowrap;overflow-x:auto}
 #params b{color:#e8e8ea} #params .d{color:#6b7280} #params .g{color:#4b5563}
 #params .v{color:#7aa2f7;font-weight:600}
 .wrap{display:flex;justify-content:center;padding:10px 10px 4px}
 img.main{height:34vh;width:auto;max-width:98vw;object-fit:contain;image-rendering:pixelated;background:#000;border:1px solid #222}
 #bar{display:flex;align-items:center;gap:10px;justify-content:center;padding:6px 12px}
 #bar button{background:#1a1a1e;color:#d4d4d8;border:1px solid #2a2a30;border-radius:5px;padding:3px 9px;cursor:pointer;font-family:inherit}
 #bar button:hover{background:#26262c} #bar button.on{background:#1e3a8a;border-color:#3b82f6;color:#fff}
 #seek{flex:0 1 420px} #posn{min-width:64px;text-align:right;color:#9ca3af}
 .sec{padding:6px 14px 2px} .sec .lbl{color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:3px}
 .sec img{max-width:98vw;image-rendering:pixelated;background:#000;border:1px solid #1c1c20}
 .legend{margin-top:4px;color:#9ca3af;font-size:11px} .legend .b{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px;vertical-align:-1px}
 /* legend colors MUST match _ROLE_COLOR (vis_server.py L68-70): context red, executed blue, lookahead green */
 .legend .b.ctx{background:#f55050} .legend .b.exe{background:#5096ff} .legend .b.look{background:#46c878}
 #dbar{display:flex;align-items:center;gap:8px;justify-content:center;padding:4px 12px;flex-wrap:wrap}
 #dbar button{background:#1a1a1e;color:#d4d4d8;border:1px solid #2a2a30;border-radius:5px;padding:2px 8px;cursor:pointer;font-family:inherit;font-size:12px}
 #dbar button:hover{background:#26262c} #dbar button.on{background:#1e3a8a;border-color:#3b82f6;color:#fff}
 #dseek{flex:0 1 320px} #dsel{min-width:120px;text-align:center;color:#9ca3af} #dposn{min-width:88px;text-align:right;color:#9ca3af}
 #perf{padding:6px 14px;border-bottom:1px solid #222;color:#9ca3af;font-size:12px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;white-space:nowrap}
 #perf b{color:#e8e8ea;font-weight:600} #perf .k{color:#6b7280} #perf .dot{color:#3a3a42}
 #perf .ok{color:#34d399} #perf svg{vertical-align:middle}
</style></head>
<body>
 <div id="params">vera viewer — connecting…</div>
 <div id="perf">⚡ measuring inference…</div>
 <div class="wrap"><img id="vis" class="main" src="/policy.mjpg" alt="current | dream+tracks | jacobian"/></div>
 <div id="bar">
   <button id="live" class="on">● LIVE</button>
   <button id="prev">◀</button><button id="pp">▶</button><button id="next">▶▶</button>
   <input type="range" id="seek" min="0" max="0" value="0"/>
   <span id="posn">LIVE</span>
 </div>
 <div class="sec"><div class="lbl">dream rollout &mdash; per-chunk · context · executed · lookahead</div>
   <div class="wrap"><img id="dream" class="main" alt="dream timeline"/></div>
   <div id="dbar">
     <button id="dfollow" class="on">● LATEST</button>
     <button id="ddprev">◀ chunk</button><span id="dsel">–</span><button id="ddnext">chunk ▶</button>
     <button id="dpp">⏸</button>
     <input type="range" id="dseek" min="0" max="0" value="0"/>
     <span id="dposn">–</span>
   </div>
   <div class="legend"><span class="b ctx"></span>context<span class="b exe"></span>executed<span class="b look"></span>lookahead (not run)</div>
 </div>
<script>
const $=function(i){return document.getElementById(i);};
let live=true, playing=false, pos=0, total=1, loading=false;
let dreams=[], dSelId=null, dPos=0, dPlaying=true, dLoading=false;  // dSelId=null => follow latest
const vis=$('vis'), seek=$('seek');
const BLANK='data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==';
function fmt(v){return (v==null||v==='')?'–':v;}
function sparkline(arr,w,h){ if(!arr||arr.length<2) return '';
  const mn=Math.min.apply(null,arr), mx=Math.max.apply(null,arr), rng=(mx-mn)||1, n=arr.length;
  let pts=''; for(let i=0;i<n;i++){ const x=(i/(n-1))*(w-3)+1.5; const y=h-1.5-((arr[i]-mn)/rng)*(h-3); pts+=x.toFixed(1)+','+y.toFixed(1)+' '; }
  const ly=(h-1.5-((arr[n-1]-mn)/rng)*(h-3)).toFixed(1);
  return '<svg width="'+w+'" height="'+h+'"><polyline points="'+pts.trim()+'" fill="none" stroke="#7aa2f7" stroke-width="1.4"/>'+
    '<circle cx="'+(w-1.5)+'" cy="'+ly+'" r="2" fill="#7aa2f7"/></svg>'; }
// onload-chained frame load: the play clock only advances once the previous frame has actually
// loaded (see the interval below). Without this, a busy server (16B inference holds the vis lock)
// returns frames slower than the clock ticks, so each src= cancels the prior in-flight load and the
// image looks frozen while the counter races ahead. The `loading` latch fixes exactly that.
vis.addEventListener('load',function(){ loading=false; });
vis.addEventListener('error',function(){ loading=false; });
function showFrame(p){ pos=Math.max(0,Math.min(total-1,p)); seek.value=pos;
  $('posn').textContent=(pos+1)+'/'+total; loading=true;
  vis.src='/policy_frame.jpg?pos='+pos+'&t='+Date.now(); }
function killStream(){ if(String(vis.src).indexOf('policy.mjpg')>=0){ vis.src=BLANK; } }
function setLive(on){ live=on; $('live').classList.toggle('on',on);
  if(on){ playing=false; $('pp').textContent='▶'; $('posn').textContent='LIVE'; loading=false; vis.src='/policy.mjpg?'+Date.now(); }
  else { killStream(); showFrame(pos); } }
function seekTo(v){ setLive(false); showFrame(parseInt(v)); }
seek.addEventListener('input',function(){ seekTo(seek.value); });
seek.addEventListener('change',function(){ seekTo(seek.value); });   // click-on-track -> jump to frame
$('prev').onclick=function(){ setLive(false); showFrame(pos-1); };
$('next').onclick=function(){ setLive(false); showFrame(pos+1); };
$('live').onclick=function(){ setLive(true); };
document.addEventListener('keydown',function(e){
  if(e.key==='ArrowLeft'){ setLive(false); showFrame(pos-1); e.preventDefault(); }
  else if(e.key==='ArrowRight'){ setLive(false); showFrame(pos+1); e.preventDefault(); }
  else if(e.key===' '){ $('pp').click(); e.preventDefault(); } });
$('pp').onclick=function(){ playing=!playing; $('pp').textContent=playing?'⏸':'▶'; if(playing){ setLive(false);} };
setInterval(function(){ if(playing && !live && !loading){
  if(pos>=total-1){ playing=false; $('pp').textContent='▶'; } else showFrame(pos+1); } },120);
async function poll(){ try{
  const s=await (await fetch('/stats.json',{cache:'no-store'})).json();
  total=Math.max(1, s.video_len||1); seek.max=total-1;  /* seek indexes _video_frames, not infer-call count */
  if(live){ seek.value=total-1; }
  const m=s.metadata||{};
  const tc=(m.teacache_thresh==null)?'off':m.teacache_thresh;
  const hist=(s.infer_ms_history||[]).filter(function(x){return x>0;});
  try{ if(hist.length){ const lastS=(s.last_infer_ms||hist[hist.length-1])/1000;
    const avgS=hist.reduce(function(a,b){return a+b;},0)/hist.length/1000;
    $('perf').innerHTML='<span class=k>⚡ infer</span> <b>'+lastS.toFixed(1)+'s</b>/chunk'+
      ' <span class=dot>•</span> <span class=k>avg</span> '+avgS.toFixed(1)+'s'+
      ' <span class=dot>•</span> '+(parseInt(fmt(m.action_horizon))/lastS).toFixed(2)+' env-steps/s'+
      ' &nbsp;'+sparkline(hist.slice(-40),130,20)+'&nbsp;'+
      ' <span class=dot>•</span> <span class=ok>flash v2</span> <span class=k>·</span> teacache '+tc+
      ' <span class=k>·</span> '+fmt(m.sample_steps)+' steps'; } }catch(ep){ $('perf').textContent='⚡ perf unavailable'; }
  try{ const dj=await (await fetch('/dream_rollouts.json',{cache:'no-store'})).json(); dreams=dj.dream_rollouts||[]; }catch(e2){}
  $('params').innerHTML='<b>vera</b> · <span class=v>'+fmt(m.embodiment)+'</span> &nbsp; <span class=d>WAN</span> '+fmt(m.planner_model)+' · <span class=d>IDM</span> '+fmt(m.idm_model)+' &nbsp;|&nbsp; H='+fmt(m.action_horizon)+' · '+fmt(m.control_hz)+'Hz · '+fmt(m.action_space)+' · '+fmt(m.sample_steps)+' steps · teacache '+tc+' &nbsp; <span class=g>'+fmt(m.git)+'</span>';
}catch(e){ $('params').textContent='waiting for server…'; } }
setInterval(poll,1000); poll();
// Dream rollout as a LOOPING VIDEO of the LATEST dream only (not a growing stack of stills): cycle
// frame_pos 0..dreamLen-1 through context->executed->lookahead (role-colored border). onload-chained
// like the main canvas so a busy server never leaves it frozen.
// ---- dream rollout mini-player: per-chunk selector + scrub bar (its own play bar) ----
// Each buffered dream (up to 8) is selectable by STABLE record_id (not deque index, which shifts as
// new dreams arrive). dSelId=null follows the latest. Frame role (context/executed/lookahead) is
// derived from the dream's own context_len/exec_horizon so the counter label matches the box border.
const dreamImg=$('dream');
dreamImg.addEventListener('load',function(){ dLoading=false; });
dreamImg.addEventListener('error',function(){ dLoading=false; });
function dIdx(){ if(dSelId!=null){ for(let i=0;i<dreams.length;i++){ if(dreams[i].record_id===dSelId) return i; } }
  return dreams.length-1; }                     // follow latest (or selected rolled out of buffer)
function dCur(){ const i=dIdx(); return (i>=0 && i<dreams.length)?dreams[i]:null; }
function dShow(p){ const d=dCur(); if(!d) return; const tot=Math.max(1,d.total_len||1);
  dPos=((p%tot)+tot)%tot; const sk=$('dseek'); sk.max=tot-1; sk.value=dPos;
  const cl=d.context_len||0, eh=d.exec_horizon||0;
  const role = dPos<cl ? 'context' : (dPos<cl+eh ? 'executed' : 'lookahead');
  $('dposn').textContent=(dPos+1)+'/'+tot+' · '+role;
  $('dsel').textContent='chunk '+(d.dream_index==null?'?':d.dream_index)+' ('+(dIdx()+1)+'/'+dreams.length+')';
  dLoading=true; dreamImg.src='/dream_rollouts_frame.jpg?id='+d.record_id+'&pos='+dPos+'&t='+Date.now(); }
function dFollow(on){ $('dfollow').classList.toggle('on',on); if(on){ dSelId=null; dPlaying=true; $('dpp').textContent='⏸'; } }
$('dfollow').onclick=function(){ dFollow(true); dShow(dPos); };
$('ddprev').onclick=function(){ const i=dIdx(); if(i>0){ dSelId=dreams[i-1].record_id; $('dfollow').classList.remove('on'); dShow(0);} };
$('ddnext').onclick=function(){ const i=dIdx(); if(i>=0 && i<dreams.length-1){ dSelId=dreams[i+1].record_id; $('dfollow').classList.remove('on'); dShow(0);} };
$('dpp').onclick=function(){ dPlaying=!dPlaying; $('dpp').textContent=dPlaying?'⏸':'▶'; };
function dSeekEv(){ dPlaying=false; $('dpp').textContent='▶'; dShow(parseInt($('dseek').value)); }
$('dseek').addEventListener('input',dSeekEv); $('dseek').addEventListener('change',dSeekEv);
setInterval(function(){ if(dPlaying && !dLoading && dreams.length){ dShow(dPos+1); } },180);
setInterval(function(){ if(live && vis.naturalWidth===0){ vis.src='/policy.mjpg?'+Date.now(); } },5000);
</script>
</body></html>
"""


class _VisHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler_cls, hub: VisHub) -> None:
        super().__init__(addr, handler_cls)
        self.hub = hub


def _make_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            return

        @property
        def hub(self) -> VisHub:
            return self.server.hub  # type: ignore[attr-defined]

        def do_GET(self):  # noqa: N802
            from urllib.parse import urlparse, parse_qs

            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                self._send_html(_DASHBOARD_HTML)
            elif path == "/stats.json":
                self._send_json(self.hub.snapshot_stats())
            elif path == "/history.json":
                self._send_json({"history": self.hub.snapshot_history()})
            elif path == "/dream_rollouts.json":
                self._send_json({"dream_rollouts": self.hub.snapshot_dream_rollout_summaries()})
            elif path == "/policy.mjpg":
                self._stream_mjpg(self.hub.snapshot_policy_jpeg)
            elif path == "/policy.jpg":
                self._send_jpeg(self.hub.snapshot_policy_jpeg())
            elif path == "/policy_frame.jpg":
                pos = self._int_arg(query, "pos", default=0)
                if pos is None:
                    return
                self._send_jpeg(self.hub.snapshot_policy_frame_jpeg(pos))
            elif path == "/input.mjpg":
                self._stream_mjpg(self.hub.snapshot_input_jpeg)
            elif path == "/input.jpg":
                self._send_jpeg(self.hub.snapshot_input_jpeg())
            elif path == "/input_frame.jpg":
                pos = self._int_arg(query, "pos", default=0)
                if pos is None:
                    return
                self._send_jpeg(self.hub.snapshot_input_frame_jpeg(pos))
            elif path == "/input_live.mjpg":
                self._stream_mjpg(self.hub.snapshot_input_live_jpeg)
            elif path == "/input_live.jpg":
                self._send_jpeg(self.hub.snapshot_input_live_jpeg())
            elif path == "/queue.mjpg":
                stream = (query.get("stream", ["policy"]) or ["policy"])[0]
                self._stream_mjpg(lambda: self.hub.snapshot_strip_jpeg(stream))
            elif path == "/queue.jpg":
                stream = (query.get("stream", ["policy"]) or ["policy"])[0]
                self._send_jpeg(self.hub.snapshot_strip_jpeg(stream))
            elif path == "/flow_diff.mjpg":
                self._stream_mjpg(self.hub.snapshot_flow_diff_jpeg)
            elif path == "/flow_diff.jpg":
                pos = self._int_arg(query, "pos", default=-1)
                if pos is None:
                    return
                self._send_jpeg(
                    self.hub.snapshot_flow_diff_jpeg(None if pos < 0 else pos)
                )
            elif path == "/dream_rollouts.mjpg":
                self._stream_mjpg(self.hub.snapshot_dream_rollouts_jpeg)
            elif path == "/dream_rollouts.jpg":
                self._send_jpeg(self.hub.snapshot_dream_rollouts_jpeg())
            elif path == "/dream_rollouts_frame.jpg":
                pos = self._int_arg(query, "pos", default=0)
                if pos is None:
                    return
                row = None
                record_id = None
                if "row" in query:
                    row = self._int_arg(query, "row")
                    if row is None:
                        return
                if "id" in query:
                    record_id = self._int_arg(query, "id")
                    if record_id is None:
                        return
                self._send_jpeg(
                    self.hub.snapshot_dream_rollouts_jpeg(
                        pos,
                        row_index=row,
                        record_id=record_id,
                    )
                )
            elif path == "/video/pause":
                self.hub.video_set_paused(True)
                self._send_json({"ok": True, "paused": True})
            elif path == "/video/play":
                self.hub.video_set_paused(False)
                self._send_json({"ok": True, "paused": False})
            elif path == "/video/seek":
                pos = self._int_arg(query, "pos")
                if pos is None:
                    return
                self.hub.video_seek(pos)
                self._send_json({"ok": True, "cursor": pos})
            elif path == "/video/step":
                delta = self._int_arg(query, "delta", default=1)
                if delta is None:
                    return
                self.hub.video_step(delta)
                self._send_json({"ok": True, "delta": delta})
            elif path == "/video/rate":
                rate = self._float_arg(query, "rate", default=1.0)
                if rate is None:
                    return
                self.hub.video_set_rate(rate)
                self._send_json({"ok": True, "rate": rate})
            elif path == "/video/live":
                self.hub.video_jump_to_live()
                self._send_json({"ok": True})
            elif path == "/obs/pause":
                self.hub.obs_set_paused(True)
                self._send_json({"ok": True, "paused": True})
            elif path == "/obs/play":
                self.hub.obs_set_paused(False)
                self._send_json({"ok": True, "paused": False})
            elif path == "/obs/seek":
                pos = self._int_arg(query, "pos")
                if pos is None:
                    return
                self.hub.obs_seek(pos)
                self._send_json({"ok": True, "cursor": pos})
            elif path == "/obs/step":
                delta = self._int_arg(query, "delta", default=1)
                if delta is None:
                    return
                self.hub.obs_step(delta)
                self._send_json({"ok": True, "delta": delta})
            elif path == "/obs/rate":
                rate = self._float_arg(query, "rate", default=1.0)
                if rate is None:
                    return
                self.hub.obs_set_rate(rate)
                self._send_json({"ok": True, "rate": rate})
            elif path == "/obs/live":
                self.hub.obs_jump_to_live()
                self._send_json({"ok": True})
            elif path == "/obs/replay_tail":
                frames = self._int_arg(query, "frames", default=60)
                if frames is None:
                    return
                cursor = self.hub.obs_replay_tail(frames)
                self._send_json({"ok": True, "cursor": cursor, "frames": frames})
            elif path == "/sync":
                on = (query.get("on", ["1"]) or ["1"])[0] in ("1", "true", "True")
                self.hub.sync_set(on)
                self._send_json({"ok": True, "sync_cursors": on})
            else:
                self.send_error(404, "not found")

        # ---------- helpers ----------

        def _int_arg(self, query: dict, key: str, default: int | None = None) -> int | None:
            try:
                vals = query.get(key)
                if vals is None:
                    if default is not None:
                        return default
                    self.send_error(400, f"missing {key}")
                    return None
                return int(vals[0])
            except (ValueError, IndexError):
                self.send_error(400, f"bad {key}")
                return None

        def _float_arg(self, query: dict, key: str, default: float | None = None) -> float | None:
            try:
                vals = query.get(key)
                if vals is None:
                    if default is not None:
                        return default
                    self.send_error(400, f"missing {key}")
                    return None
                return float(vals[0])
            except (ValueError, IndexError):
                self.send_error(400, f"bad {key}")
                return None

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_jpeg(self, jpeg: bytes | None) -> None:
            body = jpeg or _placeholder_jpeg()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, private")
            self.end_headers()
            self.wfile.write(body)

        def _stream_mjpg(self, get_jpeg) -> None:
            self.send_response(200)
            self.send_header("Cache-Control", "no-store, private")
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
            )
            self.end_headers()
            period = 1.0 / _STREAM_FPS
            placeholder = _placeholder_jpeg()
            try:
                while True:
                    jpeg = get_jpeg() or placeholder
                    self.wfile.write(f"--{_BOUNDARY}\r\n".encode("ascii"))
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    time.sleep(period)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


_PLACEHOLDER_CACHE: bytes | None = None


def _placeholder_jpeg() -> bytes:
    global _PLACEHOLDER_CACHE
    if _PLACEHOLDER_CACHE is not None:
        return _PLACEHOLDER_CACHE
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    cv2.putText(
        img, "waiting for frames...", (18, 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA,
    )
    _PLACEHOLDER_CACHE = _encode_jpeg(img[..., ::-1]) or b""
    return _PLACEHOLDER_CACHE


def start_vis_server(
    hub: VisHub, host: str = "0.0.0.0", port: int = 8766
) -> _VisHTTPServer:
    handler_cls = _make_handler()
    server = _VisHTTPServer((host, port), handler_cls, hub)
    thread = threading.Thread(
        target=server.serve_forever, name="okto-vis-http", daemon=True
    )
    thread.start()
    logger.info(
        "Vis dashboard live at http://%s:%d/ (policy.mjpg, input.mjpg, flow_diff.mjpg)",
        host if host != "0.0.0.0" else "localhost",
        port,
    )
    return server
