# How to use `link/video_pipeline.py`

First, the thing that will make your life easiest is to **export** your checkpoints using `scripts/export_model.py`.

Example command to run: You have to specify your two directories with which you trained your decoder and pixel models.
```bash
python scripts/export_model.py \
    --exp_dir outputs/2026-03-04/18-15-05 \
    --export_root exported_models \
    --name allegro_t2v \
    --flow_decoder_exp_dir outputs/2026-03-04/15-01-31 \
    --ckpt_mode symlink   # or "strip" to copy and remove optimizer states
```
Choose `symlink` if you're just exporting within your file system so that you don't recopy all the files. choose `strip` if you want to move this around at all (gets rid of optimizer states, etc.) Afterwards, you should find a neat little directory with the following:
```
exported_models/allegro_t2v/
├── algo_config.yaml
├── flow_decoder.ckpt
└── video_model.ckpt
```
Then, to use the video pipeline, simply do:

```python
from link.wan_pipeline import GenerationConfig, VideoCondition, WanPipeline

# Load from exported model (uses algo_config.yaml + flow_decoder.ckpt + video_model.ckpt)
wrapper = WanPipeline.from_config("exported_models/allegro_t2v/algo_config.yaml")

# Minimal generation: context frames [B, T, 3, H, W] in [-1, 1] + optional text
ctx_frames = ...  # [1, required_pixel, 3, H, W] in [-1, 1]; required_pixel = 1 + (N-1)*stride. if you don't match this it will be rejected
condition = VideoCondition(context_frames=ctx_frames, text="")
out = wrapper.generate(condition, GenerationConfig(decode_outputs=["rgb", "flow", "flow_rgb"]))
# out["rgb"] → [1, T, 3, H, W] in [-1, 1]
```