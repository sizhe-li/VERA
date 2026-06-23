"""
link/wan_pipeline.py

WanPipeline: concrete inference pipeline for WanTextToVideo / WanImageToVideo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf, open_dict


@dataclass
class MotionTrackConfig:
    """Controls optional motion-tracker post-processing on generated RGB videos."""

    backend: str = "alltracker"
    enabled: bool = True
    return_visualization: bool = True
    chunk_size: int | None = None
    rate: int = 2
    query_frame: int = 0
    inference_iters: int = 4
    conf_thr: float = 0.60
    bkg_opacity: float = 0.0
    temporal_stride: int = 1
    cotracker_model_name: str = "cotracker3_offline"
    cotracker_grid_size: int = 15


from vera.video_model.link.pipeline_base import (
    BaseVideoPipeline,
    GenerationConfig as PipelineGenerationConfig,
    VideoCondition,
)


@dataclass
class GenerationConfig(PipelineGenerationConfig):
    # WAN/okto tracking consumers pass view metadata through GenerationConfig.
    view_keys: list[str] | None = None
    view_widths: list[int] | None = None


def _use_fsdp() -> bool:
    """True when multi-GPU distributed inference should shard WAN."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return False
    return torch.distributed.get_world_size() > 1


class WanPipeline(BaseVideoPipeline):
    """
    Inference pipeline for WanTextToVideo / WanImageToVideo.

    Handles model instantiation from config, device placement, and exposes
    autoregressive generation: context latents → M future frames via sample_seq.

    Usage
    -----
    ::

        from omegaconf import OmegaConf
        from vera.video_model.link.wan_pipeline import WanPipeline, VideoCondition, GenerationConfig

        cfg = OmegaConf.load("configurations/algorithm/wan_t2v.yaml")
        wrapper = WanPipeline(cfg, device="cuda:0")

        ctx_frames = torch.zeros(1, 13, 3, 480, 832)  # [B, T, C, H, W] in [-1,1]
        condition = VideoCondition(context_frames=ctx_frames, text="a robot arm picks up a mug")
        out = pipeline.generate(condition, GenerationConfig(decode_outputs=["rgb", "flow_rgb"]))
        # out["rgb"]: [1, T, 3, 480, 832] in [-1, 1]
    """

    def __init__(
        self,
        cfg: DictConfig,
        ckpt_path: str | None = None,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._cfg = cfg
        self._device = torch.device(device)
        self._dtype = dtype
        self._is_fsdp = False
        self._x_shape: list[int] | None = None

        if ckpt_path is not None:
            with open_dict(cfg):
                cfg.model.tuned_ckpt_path = ckpt_path

        # Policy / notebook inference always uses empty prompt embeds unless text is
        # explicitly provided, so skip loading the heavyweight text encoder by default.
        with open_dict(cfg):
            cfg.skip_text_encoder = bool(cfg.get("skip_text_encoder", True))

        # Resolve model class from config
        model_type = cfg.get("model", {}).get("model_type", "t2v")
        if model_type == "i2v":
            from vera.video_model.algorithms.wan.wan_i2v import WanImageToVideo

            algo_cls = WanImageToVideo
        else:
            from vera.video_model.algorithms.wan.wan_t2v import WanTextToVideo

            algo_cls = WanTextToVideo

        if _use_fsdp() and cfg.get("model") and cfg.model.get("ckpt_path"):
            cfg_fsdp = cast(
                DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            )
            with open_dict(cfg_fsdp):
                cfg_fsdp.model.build_on_meta_for_fsdp_inference = True
            self._model = algo_cls(cfg_fsdp)
            self._model.configure_model()
            self._wrap_model_fsdp()
            self._is_fsdp = True
        else:
            self._model = algo_cls(cfg)
            # Lightning trainer is None → is_inference == True → bf16 weights loaded
            self._model.configure_model()
            self._model.to(self._device, self._dtype)
            self._model.device = self._device  # custom setter on WanTextToVideo
        self._model.eval()

        height = int(getattr(cfg, "height", cfg.get("height", 256)))
        width = int(getattr(cfg, "width", cfg.get("width", 256)))
        self._x_shape = [5, height, width]
        with open_dict(cfg):
            cfg.x_shape = list(self._x_shape)

    def _wrap_model_fsdp(self) -> None:
        """Wrap the WAN diffusion model with FSDP for multi-GPU inference."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        wrap_policy = ModuleWrapPolicy(type(self._model).classes_to_shard())
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        rank = torch.distributed.get_rank()
        local_device_id = 0 if torch.cuda.device_count() == 1 else rank
        device = torch.device("cuda", local_device_id)

        diffusion_model = self._model.model
        self._model.model = torch.nn.Identity()
        self._model = self._model.to(device)
        self._model.model = FSDP(
            cast(torch.nn.Module, diffusion_model),
            auto_wrap_policy=wrap_policy,
            mixed_precision=mixed_precision,
            device_id=local_device_id,
            use_orig_params=True,
        )
        if (
            getattr(self._model, "_flow_decoder_container", None)
            and self._model._flow_decoder_container
        ):
            self._model._flow_decoder_container[0] = (
                self._model._flow_decoder_container[0].to(device)
            )
        self._device = device
        self._model.device = device

    @classmethod
    def from_config(
        cls,
        config_path: str,
        ckpt_path: str | None = None,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "WanPipeline":
        cfg = OmegaConf.load(config_path)
        return cls(cfg, ckpt_path=ckpt_path, device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    @property
    def n_context_frames(self) -> int:
        return self.required_pixel_frames

    @property
    def required_pixel_frames(self) -> int:
        return 1 + (self._model.N - 1) * self._model.vae_stride[0]

    @property
    def future_pixel_frames(self) -> int:
        return self._model.M * self._model.vae_stride[0]

    @property
    def total_pixel_frames(self) -> int:
        return self.required_pixel_frames + self.future_pixel_frames

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        if self._is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType

            with FSDP.state_dict_type(
                cast(torch.nn.Module, self._model.model), StateDictType.FULL_STATE_DICT
            ):
                return self._model.load_state_dict(state_dict, strict=strict)
        return self._model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self._model.state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        if self._is_fsdp:
            device = args[0] if args else kwargs.get("device")
            if device is not None:
                device = torch.device(device)
                fsdp_model = self._model.model
                self._model.model = torch.nn.Identity()
                self._model = self._model.to(device)
                self._model.model = fsdp_model
                if (
                    getattr(self._model, "_flow_decoder_container", None)
                    and self._model._flow_decoder_container
                ):
                    self._model._flow_decoder_container[0] = (
                        self._model._flow_decoder_container[0].to(device)
                    )
                self._device = device
                self._model.device = device
            return self
        self._model = self._model.to(*args, **kwargs)
        device = args[0] if args else kwargs.get("device")
        if device is not None:
            self._device = torch.device(device)
            self._model.device = self._device
        dtype = args[1] if len(args) > 1 else kwargs.get("dtype")
        if dtype is not None:
            self._dtype = dtype
        return self

    @staticmethod
    def _rgb_to_wan_range(rgb: torch.Tensor) -> torch.Tensor:
        return rgb * 2.0 - 1.0

    @staticmethod
    def _rgb_from_wan_range(rgb: torch.Tensor) -> torch.Tensor:
        return (rgb + 1.0) * 0.5

    @torch.no_grad()
    def generate(
        self,
        condition: VideoCondition,
        config: GenerationConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        if config is None:
            config = GenerationConfig()
        return self._generate(condition, config)

    def _get_prompt_embeds(self, text: str | list[str] | None, batch_size: int) -> list:
        """Return prompt embeddings regardless of whether the text encoder is active."""
        if self._model.skip_text_encoder or text is None:
            dim = self._model.cfg.text_encoder.text_dim
            return [
                torch.zeros(1, dim, device=self.device, dtype=self.dtype)
                for _ in range(batch_size)
            ]
        prompts = [text] * batch_size if isinstance(text, str) else list(text)
        return self._model.encode_text(prompts)

    def _resolve_sampling_path(
        self,
        text: str | list[str] | None,
    ) -> tuple[Any, str]:
        has_text_conditioning = text is not None and not bool(
            self._model.skip_text_encoder
        )
        # sample_seq runs the CFG guidance branches (3 DiT passes/step, ~13x slower
        # measured); sample_seq_v2 conditions on prompt_embeds in a single pass.
        # Route to the guided path only when guidance scales are actually nonzero —
        # a prompt alone should not buy the slow path.
        needs_guidance = bool(getattr(self._model, "lang_guidance", 0)) or bool(
            getattr(self._model, "hist_guidance", 0)
        )
        sample_seq_v2 = getattr(self._model, "sample_seq_v2", None)
        if has_text_conditioning and (needs_guidance or sample_seq_v2 is None):
            return self._model.sample_seq, "sample_seq"
        if sample_seq_v2 is not None:
            return sample_seq_v2, "sample_seq_v2"
        return self._model.sample_seq, "sample_seq"

    @torch.no_grad()
    def _generate(
        self,
        condition: VideoCondition,
        generation_config: GenerationConfig,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive generation: encode context_frames → sample M future frames.

        context_frames must contain exactly 1 + (N-1)*vae_stride[0] pixel frames
        so that encoding yields exactly N latent frames.
        """
        required_pixel = self.required_pixel_frames
        N, M = self._model.N, self._model.M
        stride = self._model.vae_stride[0]

        ctx = condition.context_frames
        if ctx is None:
            raise ValueError("context_frames required for generation")

        T_ctx = ctx.shape[1]
        if (T_ctx - 1) % stride != 0:
            raise ValueError(
                f"context_frames must satisfy (T-1) % stride == 0 "
                f"(stride={stride}), got T={T_ctx}"
            )
        K_ctx = (T_ctx - 1) // stride + 1  # number of context latent frames
        if K_ctx > N:
            raise ValueError(
                f"context_frames encodes {K_ctx} latent frames but model max is N={N}. "
                f"Pass at most {required_pixel} pixel frames."
            )

        B = ctx.shape[0]
        ctx = ctx.to(self.device, self.dtype)

        ctx_lat = self._model.encode_video(rearrange(ctx, "b t c h w -> b c t h w"))
        prompt_embeds = self._get_prompt_embeds(condition.text, B)

        clip_embeds = None
        image_embeds = None
        if getattr(self._model.cfg.model, "model_type", "t2v") == "i2v":
            clip_embeds = self._model.clip_features(ctx[:, :1])
            _, _, K_lat, h_lat, w_lat = ctx_lat.shape
            image_embeds = torch.zeros(
                B,
                4 + self._model.lat_c,
                K_lat + self._model.M,
                h_lat,
                w_lat,
                device=self.device,
                dtype=self.dtype,
            )

        sample_seq, _ = self._resolve_sampling_path(condition.text)
        future_lat = sample_seq(
            ctx_lat,
            prompt_embeds,
            clip_embeds=clip_embeds,
            image_embeds=image_embeds,
        )  # [B, C, M, H, W]

        full_outputs = self._decode(ctx_lat, future_lat, generation_config)
        return full_outputs

    @torch.no_grad()
    def _decode(
        self,
        ctx_lat: torch.Tensor,
        future_lat: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> dict[str, torch.Tensor]:
        """Decode context + future latents and package results."""
        full_lat = torch.cat([ctx_lat, future_lat], dim=2)  # [B, C, N+M, H, W]

        _valid_keys = set(self._model.decode_outputs_cfg) | {"latents"}
        invalid = set(generation_config.decode_outputs) - _valid_keys
        assert (
            not invalid
        ), f"Unknown decode_outputs keys: {invalid}. Valid keys: {_valid_keys}"

        flow_requested = {"flow", "flow_rgb"} & set(generation_config.decode_outputs)
        assert not (flow_requested and self._model.flow_decoder_vae is None), (
            f"decode_outputs includes {flow_requested} but the flow decoder is not loaded. "
            "Enable it with algorithm.flow_decoder.enabled=true and set a valid checkpoint."
        )

        decode_keys = [k for k in generation_config.decode_outputs if k != "latents"]
        result = {}
        if decode_keys:
            decoded = self._model.decode_latents(full_lat, decode_outputs=decode_keys)
            for k, v in decoded.items():
                result[k] = v.cpu()
        if "latents" in generation_config.decode_outputs:
            result["latents"] = full_lat.cpu()

        return result


class WanAllTrackerPipeline(WanPipeline):
    """WAN pipeline that augments generated RGB with AllTracker outputs."""

    def __init__(
        self,
        cfg: DictConfig,
        ckpt_path: str | None = None,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        motion_track_config: MotionTrackConfig | None = None,
    ):
        super().__init__(cfg, ckpt_path=ckpt_path, device=device, dtype=dtype)
        self._motion_track_config = motion_track_config or MotionTrackConfig()
        self._motion_tracker = None

    @classmethod
    def from_config(
        cls,
        config_path: str,
        ckpt_path: str | None = None,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        motion_track_config: MotionTrackConfig | None = None,
    ) -> "WanAllTrackerPipeline":
        cfg = OmegaConf.load(config_path)
        return cls(
            cfg,
            ckpt_path=ckpt_path,
            device=device,
            dtype=dtype,
            motion_track_config=motion_track_config,
        )

    def _get_motion_tracker(self):
        if self._motion_tracker is not None:
            return self._motion_tracker

        from vera.policy.world_models.tracker_backends import build_motion_tracker

        self._motion_tracker = build_motion_tracker(
            self._motion_track_config,
            device=self.device,
        )
        return self._motion_tracker

    def _track_motion(
        self,
        rgb: torch.Tensor,
        *,
        return_visualization: bool,
        view_keys: list[str] | None = None,
        view_widths: list[int] | None = None,
    ):
        from vera.policy.world_models.tracker_backends import infer_multiview_tracks

        return infer_multiview_tracks(
            self._get_motion_tracker(),
            rgb,
            return_visualization=return_visualization,
            view_keys=view_keys,
            view_widths=view_widths,
        )

    @torch.no_grad()
    def generate_policy_chunk(
        self,
        context_rgb: torch.Tensor,
        horizon: int | None = None,
        *,
        include_boundary_step: bool = True,
        view_keys: list[str] | None = None,
        view_widths: list[int] | None = None,
        text: str | list[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """
        Generate one policy chunk from real RGB context frames in [0, 1].

        Returns:
            xs: [B, required_pixel + horizon, 5, H, W] where RGB is in [0, 1].
                Context frames remain real context. Future frames are imagined WAN RGB.
            record: policy extras including sparse motion tracks aligned to execution.
        """
        if context_rgb.shape[1] > self.required_pixel_frames:
            raise ValueError(
                "context_rgb must contain at most "
                f"{self.required_pixel_frames} frames, got {context_rgb.shape[1]}"
            )

        # Actual number of context pixel frames passed in. Used for all
        # downstream slicing instead of self.required_pixel_frames, which
        # is the MAX context the model was trained with — the caller may
        # pass fewer frames (any length satisfying (T-1) % vae_stride == 0).
        ctx_len = int(context_rgb.shape[1])

        requested_horizon = (
            self.future_pixel_frames if horizon is None else max(1, int(horizon))
        )
        initial_context_rgb = context_rgb.detach().float().cpu()
        _, sampling_path = self._resolve_sampling_path(text)
        text_conditioning_enabled = text is not None and not bool(
            self._model.skip_text_encoder
        )
        condition = VideoCondition(
            context_frames=self._rgb_to_wan_range(context_rgb).to(
                self.device, self.dtype
            ),
            text=text,
        )

        decode_outputs = ["rgb"]
        if self._model.flow_decoder_vae is not None:
            decode_outputs.append("flow")

        # Keep the original one-shot path for horizons that fit in a single WAN chunk.
        if requested_horizon <= self.future_pixel_frames:
            result = WanPipeline.generate(
                self,
                condition,
                GenerationConfig(decode_outputs=decode_outputs),
            )

            full_rgb = self._rgb_from_wan_range(result["rgb"]).cpu()
            future_rgb = full_rgb[:, ctx_len:]
            max_horizon = future_rgb.shape[1]
            horizon = max(1, min(requested_horizon, max_horizon))

            if "flow" in result:
                full_flow = result["flow"].cpu()
                flow_start = ctx_len - 1 if include_boundary_step else ctx_len
                flow_aligned = full_flow[:, flow_start : flow_start + horizon]
            else:
                flow_aligned = torch.zeros(
                    initial_context_rgb.shape[0],
                    horizon,
                    2,
                    initial_context_rgb.shape[-2],
                    initial_context_rgb.shape[-1],
                    dtype=initial_context_rgb.dtype,
                )
        else:
            future_rgb_chunks: list[torch.Tensor] = []
            flow_chunks: list[torch.Tensor] = []
            remaining = requested_horizon
            current_condition = condition
            first_chunk = True
            # Per-iteration context length: starts at the caller-supplied
            # ctx_len and may grow after the first AR chunk as the tail of
            # the previous result["rgb"] becomes the next context.
            chunk_ctx_len = ctx_len

            while remaining > 0:
                result = WanPipeline.generate(
                    self,
                    current_condition,
                    GenerationConfig(decode_outputs=decode_outputs),
                )
                full_rgb = self._rgb_from_wan_range(result["rgb"]).cpu()
                chunk_future_rgb = full_rgb[:, chunk_ctx_len:]
                if chunk_future_rgb.shape[1] <= 0:
                    raise RuntimeError(
                        "WAN autoregressive rollout produced no future frames"
                    )

                chunk_horizon = min(remaining, int(chunk_future_rgb.shape[1]))
                future_rgb_chunks.append(chunk_future_rgb[:, :chunk_horizon])

                if "flow" in result:
                    full_flow = result["flow"].cpu()
                    flow_start = (
                        chunk_ctx_len - 1
                        if (include_boundary_step or not first_chunk)
                        else chunk_ctx_len
                    )
                    flow_chunks.append(
                        full_flow[:, flow_start : flow_start + chunk_horizon]
                    )
                else:
                    flow_chunks.append(
                        torch.zeros(
                            initial_context_rgb.shape[0],
                            chunk_horizon,
                            2,
                            initial_context_rgb.shape[-2],
                            initial_context_rgb.shape[-1],
                            dtype=initial_context_rgb.dtype,
                        )
                    )

                remaining -= chunk_horizon
                first_chunk = False
                if remaining <= 0:
                    break

                next_ctx = result["rgb"][:, -self.required_pixel_frames :]
                current_condition = VideoCondition(
                    context_frames=next_ctx.clone(),
                    text=condition.text,
                )
                chunk_ctx_len = int(next_ctx.shape[1])

            future_rgb = torch.cat(future_rgb_chunks, dim=1)
            flow_aligned = torch.cat(flow_chunks, dim=1)
            horizon = int(future_rgb.shape[1])

        # Drop view-padding from the generated dream. An omni model concatenates the real views
        # (sum(view_widths)) then pads with black up to its canvas width (view_pad_position, default
        # right). The tracker AND the jacobian separate views geometrically by view_widths, so they
        # must see ONLY the valid views — otherwise the split lands inside the pad. Crop once here so
        # both the returned dream (xs) and the tracking input inherit the valid width.
        if view_widths is not None and len(view_widths) > 1:
            valid_width = int(sum(int(w) for w in view_widths))
            canvas_width = int(future_rgb.shape[-1])
            if 0 < valid_width < canvas_width:
                pad_position = getattr(self, "_view_pad_position", "right")
                if pad_position == "right":
                    _sl = slice(0, valid_width)
                elif pad_position == "left":
                    _sl = slice(canvas_width - valid_width, canvas_width)
                else:
                    raise NotImplementedError(f"view_pad_position={pad_position!r} not supported")
                future_rgb = future_rgb[..., _sl]
                flow_aligned = flow_aligned[..., _sl]
                initial_context_rgb = initial_context_rgb[..., _sl]

        xs = torch.cat(
            [
                torch.cat(
                    [
                        initial_context_rgb,
                        future_rgb[:, :horizon],
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        torch.zeros(
                            initial_context_rgb.shape[0],
                            ctx_len,
                            2,
                            initial_context_rgb.shape[-2],
                            initial_context_rgb.shape[-1],
                            dtype=initial_context_rgb.dtype,
                        ),
                        flow_aligned,
                    ],
                    dim=1,
                ),
            ],
            dim=2,
        )

        record: dict[str, Any] | None = {
            "planner_sampling_path": sampling_path,
            "text_conditioning": text,
            "text_conditioning_enabled": text_conditioning_enabled,
            # Number of pixel context frames actually fed to the planner
            # (variable-length; see _generate). xs has shape
            # [B, context_len + horizon, 5, H, W], so consumers should slice
            # xs[:, context_len:] to get the future portion.
            "context_len": ctx_len,
        }
        if self._motion_track_config.enabled:
            tracking_rgb_full = torch.cat(
                [initial_context_rgb[:, -1:], future_rgb[:, :horizon]], dim=1
            )
            temporal_stride = max(1, int(self._motion_track_config.temporal_stride))
            if temporal_stride > 1 and tracking_rgb_full.shape[1] > 2:
                sample_indices = list(
                    range(0, int(tracking_rgb_full.shape[1]), temporal_stride)
                )
                last_index = int(tracking_rgb_full.shape[1]) - 1
                if sample_indices[-1] != last_index:
                    sample_indices.append(last_index)
                if len(sample_indices) < 2:
                    sample_indices = [0, last_index]
                tracking_rgb = tracking_rgb_full[:, sample_indices]
            else:
                sample_indices = list(range(int(tracking_rgb_full.shape[1])))
                tracking_rgb = tracking_rgb_full

            tracker_output = self._track_motion(
                self._rgb_to_wan_range(tracking_rgb).to(self.device, self.dtype),
                return_visualization=self._motion_track_config.return_visualization,
                view_keys=view_keys,
                view_widths=view_widths,
            )
            motion_tracks = tracker_output.motion_tracks.as_policy_dict()
            source_rgb = tracking_rgb_full[:, sample_indices[:-1]]
            target_rgb = tracking_rgb_full[:, sample_indices[1:]]
            record.update(
                {
                    "motion_tracks": motion_tracks,
                    "track_source_rgb": source_rgb,
                    "track_target_rgb": target_rgb,
                    "track_temporal_stride": temporal_stride,
                    "track_sample_indices": [int(idx) for idx in sample_indices],
                }
            )
            if view_keys is not None:
                record["view_keys"] = list(view_keys)
            if view_widths is not None:
                record["view_widths"] = [int(width) for width in view_widths]
            per_view_outputs = getattr(tracker_output, "per_view_outputs", None)
            if per_view_outputs is not None:
                record["per_view_motion_tracks"] = [
                    output.motion_tracks.as_policy_dict() for output in per_view_outputs
                ]
                if self._motion_track_config.return_visualization:
                    per_view_vis = [output.visualization for output in per_view_outputs]
                    record["per_view_tracker_vis"] = per_view_vis
                    record["per_view_alltracker_vis"] = per_view_vis
            if tracker_output.visualization is not None:
                record["tracker_vis"] = tracker_output.visualization
                record["alltracker_vis"] = tracker_output.visualization

        return xs, record

    @torch.no_grad()
    def generate(
        self,
        condition: VideoCondition,
        config: GenerationConfig | None = None,
    ) -> dict[str, Any]:
        if config is None:
            config = GenerationConfig()

        requested_rgb = "rgb" in config.decode_outputs
        decode_outputs = list(config.decode_outputs)
        if "rgb" not in decode_outputs:
            decode_outputs.append("rgb")

        result = super().generate(
            condition,
            GenerationConfig(decode_outputs=decode_outputs),
        )
        if not self._motion_track_config.enabled:
            return result

        tracker_output = self._track_motion(
            result["rgb"],
            return_visualization=self._motion_track_config.return_visualization,
            view_keys=config.view_keys,
            view_widths=config.view_widths,
        )
        result["motion_tracks"] = tracker_output.motion_tracks
        if tracker_output.visualization is not None:
            result["tracker_vis"] = tracker_output.visualization
            result["alltracker_vis"] = tracker_output.visualization
        if config.view_keys is not None:
            result["view_keys"] = list(config.view_keys)
        if config.view_widths is not None:
            result["view_widths"] = [int(width) for width in config.view_widths]
        per_view_outputs = getattr(tracker_output, "per_view_outputs", None)
        if per_view_outputs is not None:
            result["per_view_motion_tracks"] = [
                output.motion_tracks for output in per_view_outputs
            ]
            if self._motion_track_config.return_visualization:
                per_view_vis = [output.visualization for output in per_view_outputs]
                result["per_view_tracker_vis"] = per_view_vis
                result["per_view_alltracker_vis"] = per_view_vis

        if not requested_rgb:
            result.pop("rgb", None)
        return result
