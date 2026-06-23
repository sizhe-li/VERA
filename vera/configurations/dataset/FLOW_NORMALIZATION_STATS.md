# Flow Normalization Reproducibility

This note records reproducible normalization commands for Jacobian datasets that should
keep physical zero centered at model zero.

## Command

```bash
PYTHONPATH=project python3 project/okto/scripts/compute_dataset_stats.py drake_allegro_srg1 \
  --flow-only \
  --flow-normalization-mode symmetric_percentile \
  --flow-abs-percentile 90 \
  --flow-space train_resolution \
  --dry-run

PYTHONPATH=project python3 project/okto/scripts/compute_dataset_stats.py droid_se3_normalized \
  --write \
  --action-normalization-mode symmetric_percentile \
  --action-abs-percentile 90 \
  --flow-normalization-mode symmetric_percentile \
  --flow-abs-percentile 90 \
  --flow-space train_resolution \
  --action-space scaled_train

PYTHONPATH=project python3 project/okto/scripts/compute_dataset_stats.py droid_se3_normalized \
  --write \
  --flow-only \
  --flow-normalization-mode symmetric_percentile \
  --flow-abs-percentile 90 \
  --flow-space train_resolution \
  --flow-window-stride 32 \
  --num-workers 8 \
  --worker-chunk-size 4
```

## Result Written To Config

Target config:

- `project/okto/configurations/dataset/drake_allegro_srg1_normalized.yaml`

Computed values for `drake_allegro_srg1_normalized.yaml`:

```yaml
flow_normalization_mode: symmetric_percentile
oflow_percentile: 90.0
oflow_abs_scale:
  - 31.52375984191896
  - 28.85662117004395
flow_normalization_space: train_resolution
```

## Caution

Do not treat scale-only flow normalization as the reference pattern for DROID Jacobian
training. For DROID, the intended contract is:

- action normalization in `scaled_train` space
- flow normalization in `train_resolution` space
- symmetric-percentile stats for both action and flow

Symmetric-percentile normalization preserves zero but does not clip tails, so normalized
values may exceed `[-1, 1]`.

The DROID `--flow-only` path now loads flow directly instead of decoding RGB first, but a
full-dataset run is still expensive because deterministic strided traversal may visit many
windows across all episodes. Increase `--flow-window-stride` or cap `--max-episodes` when
you only need approximate stats.
