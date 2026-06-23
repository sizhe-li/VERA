"""vera.video_model — WAN video planner (PORTED, Phase 1).

Ported by copy + prefix-rewrite from third_party/flow-planner:
  flow-planner/algorithms -> vera.video_model.algorithms   (wan, cogvideo, common)
  flow-planner/link        -> vera.video_model.link          (wan_pipeline, pipeline_base)
  flow-planner/utils       -> vera.video_model.utils
Absolute imports were rewritten (algorithms.* -> vera.video_model.algorithms.*, link.* ->
vera.video_model.link.*, utils.* -> vera.video_model.utils.*); relative imports unchanged. The
motion-tracker reverse dep now points at vera.policy.world_models.tracker_backends (lazy; resolves
once Phase 2 lands).

Public modules (import explicitly; they pull the heavy WAN stack -- torch/einops/transformers):
  - vera.video_model.link.wan_pipeline            : WanPipeline, GenerationConfig, VideoCondition,
                                                    MotionTrackConfig
  - vera.video_model.algorithms.wan.wan_t2v       : WanTextToVideo, _load_checkpoint_weights_only
  - vera.video_model.algorithms.wan.wan_i2v       : WanImageToVideo
  - vera.video_model.algorithms.wan.modules.model : WanModel (the DiT)

NOT ported: wan_ar -- absent from the pinned flow-planner submodule (the okto reference to
algorithms.wan_ar.video_rollout is a dangling import in the original; bring it in if/when a
flow-planner commit that contains it is pinned).

This __init__ stays import-light so `import vera.video_model` works without the heavy deps.
"""
