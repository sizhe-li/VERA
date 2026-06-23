# configurations

We use [Hydra](https://hydra.cc/docs/intro/) to manage configurations. Change/Add the yaml files in this folder
to change the default configurations. You can also override the default configurations by 
passing command line arguments.

All configurations are automatically saved in wandb run.

## Experiment Registry vs Scale Overlays

- `experiment=<name>` must be a registered experiment class name in `okto/experiments/__init__.py` (for example: `jacobian_learning`).
- Files like `experiment/jacobian_scale_m0.yaml`, `experiment/jacobian_scale_m1.yaml`, and `experiment/jacobian_scale_m2.yaml` are scale-tuning overlays, not standalone experiment classes.
- To keep one-line launches, use top-level stack configs that pin `experiment=jacobian_learning` and apply scale settings:
  - `config_jacobian_scale_m0_droid_giant`
  - `config_jacobian_scale_m1_droid_giant`
  - `config_jacobian_scale_m2_droid_giant`

## One-line Launch Examples

```bash
python -u -m okto.main --config-name config_jacobian_scale_m0_droid_giant name=jacobian_m0_run1 +wandb.tags=[scaleup,m0]
python -u -m okto.main --config-name config_jacobian_scale_m1_droid_giant name=jacobian_m1_run1 +wandb.tags=[scaleup,m1]
python -u -m okto.main --config-name config_jacobian_scale_m2_droid_giant name=jacobian_m2_run1 +wandb.tags=[scaleup,m2]
```

## Runtime Defaults Embedded in Scale Stacks

The three scale stack configs now include the common runtime overrides that were previously
passed on CLI:

- Checkpoint policy: keep only `best` + `last` (`save_top_k=1`, `save_last=true`, monitor `loss/training/total`, mode `min`).
- Checkpoint cadence: `every_n_train_steps=500`.
- Validation/logging caps:
  - `experiment.validation.batch_size=10`
  - `algorithm.logging.max_validation_batches=1`
  - `algorithm.logging.max_validation_samples=10`
  - `algorithm.logging.max_validation_views=3`
  - `algorithm.logging.max_num_videos=30`
  - `algorithm.logging.max_validation_frames=16`
- Profiling defaults live under `experiment.profiling.*` (not `algorithm.profiling.*`):
  - enabled with optimizer-step scheduling and short windows (`start_after_k_steps=20`, `warmup=2`, `active=2`, `every_k_steps=5000`, `max_windows=3`)
  - Chrome trace export to `profiles/` with per-scale prefixes.
- Model setting: `algorithm.model.freeze_backbone=true`.

## Output Storage Root

All scale stack configs now write Hydra outputs to:

- single-run: `/path/to/data/jacobian/wandb_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}_${name}`
- multirun: `/path/to/data/jacobian/wandb_runs/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}`

This keeps training artifacts out of the repo-local `outputs/` directory by default.