<h1 align="center">VERA — Turning Video Models into Generalist Robot Policies</h1>

<p align="center">
  Sizhe Lester Li<sup>*</sup>,
  Evan Kim<sup>*</sup>,
  Xingjian Bai<sup>*</sup>
</p>
<p align="center">
  Tong Zhao,
  Tao Pang,
  Max Simchowitz,
  Vincent Sitzmann
</p>

<p align="center"><sup>*</sup>equal contribution</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.27817">[Paper]</a> &nbsp;·&nbsp;
  <a href="https://vera.csail.mit.edu/">[Project Page]</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/sizhe-lester-li/VERA">[Models]</a>
</p>

https://github.com/user-attachments/assets/4d5d7325-43df-43e8-ae25-222e3b2c5417

**VERA** (**V**ideo-to-**E**mbodied **R**obot **A**ction model) is a **two-stage**, closed-loop video-to-action
policy. It leaves a video generative model **as-is** as an action-free world model that "dreams" the future,
and trains an embodiment-specific **inverse-dynamics model (IDM)** — built on the robot's **Jacobian** — to
translate that dream into actions:

1. **Video planner** (`vera.video_model` / `vera.idm.dfot`) — an action-free diffusion model that generates
   future frames from the current observation (+ optional text). **Embodiment-agnostic.**
2. **Jacobian IDM** (`vera.idm` + `vera.policy`) — a faithful, data-efficient translator from dreamed future
   to robot actions. **Embodiment-specific**, swappable without retraining the planner.

> The thesis: *decoupled video planning + faithful video-to-action translation* is a viable route to
> zero-shot, cross-embodiment robot control. **One video planner, many IDMs.**

---

## 🗺️ Release roadmap

_Last updated: **Jul 4, 2026**._

| Wave | Embodiments | Code | Checkpoints | Status |
|------|-------------|:----:|:-----------:|:------:|
| **Wave 1** — released Jun 23, 2026 | **MimicGen** (Panda, 2-block stacking) · **PushT** (planar pusher) | ✅ | ✅ | **early version available** (as of Jul 4: working on bug fixes from merging the two repos) |
| **Wave 2** — ETA ~Jul 11, 2026 | Allegro-Sim · Allegro-Real · IIWA-Sim · DROID (FR3 real) | ✅ in-tree | 🔜 | code present; checkpoints + docs coming |

This repo already contains the unified code for **all** embodiments, but Wave 1 documents and ships
checkpoints only for **MimicGen + PushT**. The cross-embodiment **OMNI** WAN planner and the DROID/Allegro
IDMs land with Wave 2. We are also working on releasing the **Allegro-hand and IIWA simulators** themselves
(as of Jul 4, 2026 — the `eval` extra currently covers only the MimicGen + PushT environments).

---

## Install

VERA targets **Python 3.11** + **PyTorch 2.6 (CUDA 12.4)**. Self-contained — no sibling repos on `sys.path`.

```bash
git clone git@github.com:sizhe-li/VERA.git && cd VERA
pip install -e ".[idm,video]"            # the two stages (IDM + video planner)
```

**Simulators** (needed to reproduce the results — install the `eval` extra):

```bash
pip install -e ".[eval]"                 # gymnasium, gym-pusht, robomimic, robosuite, mimicgen, mujoco
```

- **PushT** runs on `gym-pusht` (pulls `pymunk`), but the notebook seeds rollouts from the **original
  PushT replay buffer** `pusht_cchi_v7_replay.zarr` (the known-success initial states it indexes into).
  Grab it from the [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) release:
  ```bash
  wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
  unzip pusht.zip          # -> pusht/pusht_cchi_v7_replay.zarr
  ```
  Then point the notebook's `ZARR_PATH` at `.../pusht/pusht_cchi_v7_replay.zarr`.
- **MimicGen** runs on `robosuite` + `robomimic` + `mimicgen` (all pinned in the `eval` extra) and needs
  **MuJoCo** (pulled automatically). It also needs the task **dataset HDF5** (the initial states), e.g.
  `stack_d0.hdf5` — download the standard MimicGen datasets from
  [🤗 `amandlek/mimicgen_datasets`](https://huggingface.co/datasets/amandlek/mimicgen_datasets) (or follow
  the [MimicGen instructions](https://github.com/NVlabs/mimicgen)) and point the notebook at the file.
- **flash-attn** (WAN attention) is optional — the WAN path falls back to SDPA if absent.
- **VGGT** (the IDM visual backbone — required by **both** the MimicGen and PushT IDMs) installs
  automatically with the `idm` extra as a git dependency
  ([`facebookresearch/vggt`](https://github.com/facebookresearch/vggt)). If your environment blocks git
  installs, clone and install it manually instead:
  ```bash
  pip install "git+https://github.com/facebookresearch/vggt.git"
  # or: git clone https://github.com/facebookresearch/vggt && pip install -e vggt
  ```
  The **VGGT-1B weights** are then pulled from `facebook/VGGT-1B` on first use.

Verify:
```bash
python -c "import vera, vera.policy, vera.idm, vera.server; print('vera ok')"
```

---

## ⚡ Quickest deploy

Every embodiment runs the **same two steps**: start a policy server in one terminal, then run its client
notebook in another. The notebook drives the sim, prints the success rate, and inlines the rollout videos.

```
  Terminal 1 — server                          Jupyter — client notebook
  ┌─────────────────────────────┐   websocket  ┌──────────────────────────────┐
  │ python -m vera.server        │ ───────────▶ │  open the notebook → Run All │
  │   .start_vera_server ...      │  :8800/:8820 │  → success rate + videos     │
  └─────────────────────────────┘              └──────────────────────────────┘
```

| Task | Server flag | **Client notebook (run this)** |
|---|---|---|
| **PushT** — planar push-to-goal | `--embodiment pusht` | **`examples/pusht_dfot_stack.ipynb`** |
| **MimicGen** — 2-block stacking | `--embodiment mimicgen` | **`examples/mimicgen_stack.ipynb`** |

### PushT (DFoT planner — small, loads in seconds)

**1. Start the server** (Terminal 1):
```bash
python -m vera.server.start_vera_server --embodiment pusht --port 8820 --vis-port 8821
```
**2. Run the client:** open **`examples/pusht_dfot_stack.ipynb`** → **Run All**.

- it connects to the server, rolls out a known-success initial state, prints the success rate, and inlines
  the rollout + the composite policy-vis;
- checkpoint paths come from the `VERA_PUSHT_*` env vars (see `vera/server/start_server_pusht.py`).

### MimicGen two-block stacking (WAN planner)

**1. Point at the downloaded checkpoints, then start the server** (Terminal 1):
```bash
export VERA_WAN_CKPT_ROOT=/path/to/Wan2.1-T2V-1.3B            # frozen Wan2.1 base (text-enc + VAE)
export VERA_MIMICGEN_CKPT_DIR=./vera-ckpts/mimicgen-wan-1.3b  # specialist DiT + flow decoder
python -m vera.server.start_vera_server --embodiment mimicgen --port 8800 --vis-port 8801 \
    --algo-config $VERA_MIMICGEN_CKPT_DIR/algo_config.yaml \
    --text "A robot arm stacks one block on top of another block"
```
> Set **both** env vars before launching — the hosted `algo_config.yaml` reads the DiT + flow decoder from
> `VERA_MIMICGEN_CKPT_DIR` and the Wan2.1 base from `VERA_WAN_CKPT_ROOT`.

**2. Run the client:** open **`examples/mimicgen_stack.ipynb`** → **Run All**.

- swap pieces live via env vars on the server: `VERA_DYNAMICS_RUN_ID` (IDM checkpoint),
  `VERA_TRACKER_BACKEND`, `VERA_MOTION_PLAN_SCALE`, `VERA_N_ACTION_STEPS`.

---

## Live viewer — watch the policy think

Pass `--vis-port` to any server and open `http://localhost:<vis-port>/` for a built-in dashboard that
streams VERA's **entire two-stage pipeline live**, in one strip, as the rollout runs. The policy is
interpretable by construction — not a black box:

![VERA live viewer](docs/assets/viewer.png)

Each row is one camera view, read left → right:

| Panel | What it shows |
|---|---|
| **Current** | the robot's live observation |
| **Dream + tracks** | the video model's predicted future, with motion tracks overlaid |
| **Dream** | the decoded future frames |
| **Jacobian field** | the map that turns the dream into the next action |

The per-chunk player below scrubs each generated dream chunk frame-by-frame, so the planner's imagination
and the IDM's response sit side-by-side. The notebooks inline this same composite via `show_policy_vis()`;
snapshot it any time with `python -m vera.server.save_vis_video --output dream.mp4`.

---

## Checkpoints

Hosted on HuggingFace — `huggingface.co/sizhe-lester-li/VERA`. VERA hosts only the **trained** artifacts;
frozen upstream pieces are pulled from their original homes.

| Group | dir | what |
|---|---|---|
| **MimicGen** | `mimicgen-wan-1.3b/` | specialist WAN planner (DiT-only bf16, ~2.8 GB) + `flow_decoder.ckpt` + `algo_config.yaml` |
| | `idm-mimicgen-37oa162u/` | MimicGen Jacobian IDM (+ config sidecar) |
| **PushT** | `pusht-dfot/` | DFoT flow planner (~39 MB) + `run_config.yaml` |
| | `pusht-idm/` | PushT Jacobian IDM (~232 MB) + `config.yaml` |
| **Upstream** | `Wan-AI/Wan2.1-T2V-1.3B`, `facebook/VGGT-1B` | WAN base + IDM backbone (not re-hosted) |

**Download** (with the HuggingFace CLI — `pip install huggingface_hub`):

```bash
# (1) MimicGen + PushT only — IDM + video planner for the Wave-1 notebooks   (~3.8 GB)
hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts \
    --include "mimicgen-wan-1.3b/*" "idm-mimicgen-37oa162u/*" "pusht-dfot/*" "pusht-idm/*"

# (2) everything — also pulls the 33 GB OMNI planner + DROID IDM (Wave 2)      (~42 GB)
hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts
```

The Wave-1 download is **~3.8 GB**; the full repo is **~42 GB** (the 33 GB OMNI WAN planner dominates).
Then point the server/notebook at the downloaded paths (`--algo-config`, `VERA_PUSHT_*` / `VERA_WAN_CKPT_ROOT`).

**OMNI training data (Wave 2):** the cross-embodiment OMNI WAN planner is trained on a weighted mixture of
**Allegro-Sim + Allegro-Real + MimicGen + DROID** (each kept at native fps/aspect, black-padded to a
576-wide multiview canvas). **PushT is *not yet* in the OMNI mixture** — for now it uses its own DFoT flow
planner, and we will release a **new OMNI checkpoint that includes PushT soon**. The training config for
that 5-environment mixture already ships in this repo
(`vera/configurations/config_wan_combined_5env.yaml`).

---

## Training

Both stages train through one Hydra entry point, `python -m vera.main` — see **[TRAINING.md](TRAINING.md)**
for the full guide (data format, IDM training, WAN / OMNI video-planner finetuning, multi-GPU/FSDP).

---

## Acknowledgements

This work was supported by the National Science Foundation under Grant No. 2211259, by the Intelligence
Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC)
under 140D0423C0075, by the Amazon Science Hub, by the MIT-Google Program for Computing Innovation, by
Advanced Micro Devices, Inc. under the AMD University Program's support of the MIT Hardware Consortium, and
by a 2025 MIT Office of Research Computing and Data Seed Grant.

## License & Citation

Released under the **MIT License** (see `LICENSE`); depended-upon code retains its own license (see
`NOTICE`). VERA builds on **Wan2.1** (Apache-2.0), **VGGT** (Meta), **CLIP/open_clip** (MIT), and
**cotracker/AllTracker**; the DFoT/DiT backbones are adapted from `facebookresearch/DiT` and `NVlabs/edm2`.

```bibtex
@article{li2026turningvideomodelsgeneralist,
      title={Turning Video Models into Generalist Robot Policies}, 
      author={Sizhe Lester Li and Evan Kim and Xingjian Bai and Tong Zhao and Tao Pang and Max Simchowitz and Vincent Sitzmann},
      year={2026},
      eprint={2605.27817},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.27817}, 
}
```
