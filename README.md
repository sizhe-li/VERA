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

1. **Video planner** (`vera.video_model` / `vera.idm.dfot`)
2. **Jacobian IDM** (`vera.idm` + `vera.policy`)

---

<a id="news"></a>
## 🔥 News

**Jul 12, 2026 — DROID: language-conditioned video generation (no robot required).**
If you want to see the planner itself at work before setting up any simulator, start here: the
notebook continues the same real context frames under different language prompts, and the
generated futures differ accordingly (the executed outputs ship in the notebook, so you can
inspect them before running anything). It runs the DROID WAN planner directly — single GPU,
~60 GB VRAM (bf16); no server, no sim.

If you **do** have a DROID setup, the same notebook doubles as a **zero-shot test of whether the
checkpoint generalizes to your scene**: record short clips from your own cameras, swap them in for
the bundled ones, and inspect how well the generated futures follow your prompts in your scene —
before anything runs on the robot.

**1. Point at the downloaded checkpoints:**
```bash
export VERA_DROID_CKPT_DIR=./vera-ckpts/wan-droid-14b       # DROID WAN planner (DiT + algo_config.yaml)
export VERA_WAN14B_CKPT_ROOT=/path/to/Wan2.1-I2V-14B-480P   # frozen Wan2.1 base (text-enc + VAE + CLIP)
```
**2. Run the notebook:** open **`examples/droid_generation.ipynb`** → **Run All**.

- two experiments on the bundled sample clips (`examples/droid_demo_videos/`, three synchronized
  DROID cameras): the same prompt from five start times, and three different prompts from the same
  start frame (context frames red-bordered, generation follows);
- context length and guidance come from the checkpoint's `algo_config.yaml`.

---

## 🗺️ Release roadmap

_Last updated: **Jul 12, 2026**._

| Wave | Embodiments | Code | Checkpoints | Status |
|------|-------------|:----:|:-----------:|:------:|
| **Wave 1** — released Jun 23, 2026 | **MimicGen** (Panda, 2-block stacking) · **PushT** (planar pusher) | ✅ | ✅ | **ready** |
| **Wave 2a** — Jul 12, 2026 | **DROID video generation** (WAN planner + walkthrough notebook) | ✅ | ✅ | **ready** |
| **Wave 2b** | Allegro-Sim · Allegro-Real · IIWA-Sim · DROID policy serving (FR3 real) | ✅ | 🔜 | code present; checkpoints + docs coming |

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
  PushT replay buffer** `pusht_cchi_v7_replay.zarr` (the initial states it indexes into).
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
  Terminal 1 — server                         Jupyter — client notebook
  ┌──────────────────────────────┐              ┌──────────────────────────────┐
  │ python -m vera.server        │ ───────────▶ │ open the notebook → Run All  │
  │   .start_vera_server ...     │  :8800/:8820 │ → success rate + videos      │
  └──────────────────────────────┘              └──────────────────────────────┘
```

| Task | Server flag | **Client notebook (run this)** |
|---|---|---|
| **PushT** — planar push-to-goal | `--embodiment pusht` | **`examples/pusht_dfot_stack.ipynb`** |
| **MimicGen** — 2-block stacking | `--embodiment mimicgen` | **`examples/mimicgen_stack.ipynb`** |
| **DROID** — video generation from language | *(no server needed)* | **`examples/droid_generation.ipynb`** — setup in [🔥 News](#news) |

### PushT (DFoT planner — small, loads in seconds)

**1. Start the server** (Terminal 1):
```bash
python -m vera.server.start_vera_server --embodiment pusht --port 8820 --vis-port 8821
```
**2. Run the client:** open **`examples/pusht_dfot_stack.ipynb`** → **Run All**.

- it connects to the server, rolls out the walkthrough's default start state (a single episode — set
  `FRAME_INDICES = None` in the notebook for a population success rate), prints the result, and inlines
  the rollout + the composite policy-vis;
- checkpoint paths come from the `VERA_PUSHT_*` env vars (see `vera/server/start_server_pusht.py`);
- the server plans 3 future frames per replan and executes 2 of them (`VERA_PUSHT_ACTION_CHUNK_HORIZON=3`,
  `VERA_PUSHT_N_ACTION_STEPS=2`, both env-overridable).

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
> The Jacobian IDM checkpoint loads locally via `VERA_MIMICGEN_DYNAMICS_CKPT`
> (default: `./vera-ckpts/idm-mimicgen-285ouq1q/model.ckpt`).

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
| **PushT** | `pusht-dfot/` | DFoT flow planner (~39 MB) + `run_config.yaml` |
| | `pusht-idm/` | PushT Jacobian IDM (~232 MB) + `config.yaml` |
| **DROID** | `wan-droid-14b/` | DROID WAN planner (DiT-only bf16, ~31 GB) + `algo_config.yaml` |
| | `droid-demo-clips/` | sample multi-view robot clips for the generation walkthrough (~100 MB) |
| **Upstream** | `Wan-AI/Wan2.1-T2V-1.3B`, `Wan-AI/Wan2.1-I2V-14B-480P`, `facebook/VGGT-1B` | WAN bases + IDM backbone (not re-hosted) |

**Download** (with the HuggingFace CLI — `pip install huggingface_hub`):

```bash
# (1) MimicGen + PushT only — IDM + video planner for the Wave-1 notebooks   (~15 GB)
hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts \

# (2) everything — also pulls the 33 GB OMNI planner, DROID IDM, and the
#     31 GB DROID WAN planner for the generation walkthrough                   (~73 GB)
hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts
```

The Wave-1 download is **~15 GB (11.3 GB of it the VGGT-based MimicGen IDM)**; the full repo is **~73 GB** (the 33 GB OMNI planner and the 31 GB DROID planner dominate).
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
