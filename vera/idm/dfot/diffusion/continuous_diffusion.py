"""
A very minimal implementation of continuous-time diffusion models. For compatibility with other modules,
sampling schedules are still implemented in discrete time.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from .discrete_diffusion import DiscreteDiffusion, ModelPrediction


class ContinuousNoiseSchedule(nn.Module, ABC):
    """
    An abstract class for continuous noise schedule that is compatible with continuous-time diffusion models.
    """

    @classmethod
    def from_config(cls, cfg: DictConfig):
        match cfg.name:
            case "cosine":
                return CosineNoiseSchedule(cfg)
            case _:
                raise ValueError(f"unknown noise schedule {cfg.name}")

    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Given the timestep t within [0, 1], return the logSNR value at that timestep."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_logsnr(self) -> torch.Tensor:
        """Return the maximum logSNR value."""
        raise NotImplementedError

    @property
    @abstractmethod
    def min_logsnr(self) -> torch.Tensor:
        """Return the minimum logSNR value."""
        raise NotImplementedError


class CosineNoiseSchedule(ContinuousNoiseSchedule):
    """
    Cosine noise schedule that can be shifted from base resolution to target resolution,
    proposed in Simple Diffusion (2023, https://arxiv.org/abs/2301.11093).
    Here, `shift` should be set to `base_resolution / target_resolution`.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        logsnr_min, logsnr_max = cfg.get("logsnr_min", -15.0), cfg.get(
            "logsnr_max", 15.0
        )
        shift = cfg.get("shift", 1.0)
        self.register_buffer(
            "t_min",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max, dtype=torch.float32))),
            persistent=False,
        )
        self.register_buffer(
            "t_max",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min, dtype=torch.float32))),
            persistent=False,
        )
        self.register_buffer(
            "shift",
            2 * torch.log(torch.tensor(shift, dtype=torch.float32)),
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return (
            -2 * torch.log(torch.tan(self.t_min + t * (self.t_max - self.t_min)))
            + self.shift
        )

    @property
    def max_logsnr(self) -> torch.Tensor:
        return self.forward(
            torch.tensor(0.0, dtype=torch.float32, device=self.shift.device)
        )

    @property
    def min_logsnr(self) -> torch.Tensor:
        return self.forward(
            torch.tensor(1.0, dtype=torch.float32, device=self.shift.device)
        )


class ContinuousDiffusion(DiscreteDiffusion):
    def __init__(
        self,
        cfg: DictConfig,
        backbone_cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
    ):
        super().__init__(cfg, backbone_cfg, x_shape, max_tokens, external_cond_dim)
        assert (
            self.objective == "pred_v" and self.loss_weighting.strategy == "sigmoid"
        ), "ContinuousDiffusion only supports 'pred_v' objective and 'sigmoid' loss weighting"
        self.precond_scale = cfg.precond_scale
        self.sigmoid_bias = cfg.loss_weighting.sigmoid_bias

    def _build_buffer(self):
        super()._build_buffer()
        self.training_schedule = ContinuousNoiseSchedule.from_config(
            self.cfg.training_schedule
        )

    def model_predictions(
        self, x, k, external_cond=None, external_cond_mask=None, logsnr=None
    ):
        # print("x", x.shape, "k", k.shape, "external_cond", external_cond.shape)

        x_dim = x.shape[2]
        if external_cond is not None:
            x = torch.cat([x, external_cond], dim=2)
            external_cond = None  # already concatenated

        is_training = True
        if logsnr is None:
            logsnr = self.logsnr[k]
            # print("using testing schedule")
            # print("using training schedule")
            # logsnr = self.training_schedule(k)
            is_training = False

        model_output = self.model(
            x, self.precond_scale * logsnr, external_cond, external_cond_mask
        )

        # NOTE: trim the input-output in case we are doing conditioning
        model_output = model_output[:, :, :x_dim]
        x = x[:, :, :x_dim]

        if is_training:  # return immediately since we are in training mode
            return ModelPrediction(None, None, model_output)

        else:
            if self.objective == "pred_noise":
                pred_noise = torch.clamp(
                    model_output, -self.clip_noise, self.clip_noise
                )
                x_start = self.predict_start_from_noise(x, k, pred_noise)

            elif self.objective == "pred_x0":
                x_start = model_output
                pred_noise = self.predict_noise_from_start(x, k, x_start)

            elif self.objective == "pred_v":
                v = model_output
                x_start = self.predict_start_from_v(x, k, v)
                pred_noise = self.predict_noise_from_v(x, k, v)

            model_pred = ModelPrediction(pred_noise, x_start, model_output)

            return model_pred

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
    ):
        # add noise
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        ### continuous noise schedule
        logsnr = self.training_schedule(k)
        alpha_t = self.add_shape_channels(torch.sigmoid(logsnr).sqrt())
        sigma_t = self.add_shape_channels(torch.sigmoid(-logsnr).sqrt())

        if self.context_channel_noise_enabled and self.n_context_tokens > 0:
            c_start, c_end = (
                self.zero_noise_channel_range[0],
                self.zero_noise_channel_range[1],
            )
            n_tokens = k.shape[1]
            is_context = (
                torch.arange(n_tokens, device=k.device, dtype=k.dtype)[None, :]
                < self.n_context_tokens
            )
            k_zero_noise = torch.where(is_context, torch.zeros_like(k), k)
            logsnr_zero = self.training_schedule(k_zero_noise)
            alpha_zero = self.add_shape_channels(
                torch.sigmoid(logsnr_zero).sqrt()
            )
            sigma_zero = self.add_shape_channels(
                torch.sigmoid(-logsnr_zero).sqrt()
            )

            x_other = x[:, :, :c_start]
            x_zero = x[:, :, c_start:c_end]
            noise_other = noise[:, :, :c_start]
            noise_zero = noise[:, :, c_start:c_end]

            x_t_other = alpha_t * x_other + sigma_t * noise_other
            x_t_zero = alpha_zero * x_zero + sigma_zero * noise_zero

            if c_end < x.shape[2]:
                x_rest = x[:, :, c_end:]
                noise_rest = noise[:, :, c_end:]
                x_t_rest = alpha_t * x_rest + sigma_t * noise_rest
                x_t = torch.cat([x_t_other, x_t_zero, x_t_rest], dim=2)
            else:
                x_t = torch.cat([x_t_other, x_t_zero], dim=2)
        else:
            x_t = alpha_t * x + sigma_t * noise

        v_pred = self.model_predictions(x_t, k, external_cond, logsnr=logsnr).model_out

        noise_pred = alpha_t * v_pred + sigma_t * x_t
        x_pred = alpha_t * x_t - sigma_t * v_pred

        # For context_channel_noise: target noise is 0 for zero_noise channels in context
        if self.context_channel_noise_enabled and self.n_context_tokens > 0:
            target_noise = noise.detach().clone()
            is_context_expand = is_context.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            target_noise[:, :, c_start:c_end] = torch.where(
                is_context_expand,
                torch.zeros_like(target_noise[:, :, c_start:c_end]),
                target_noise[:, :, c_start:c_end],
            )
        else:
            target_noise = noise.detach()

        loss = F.mse_loss(noise_pred, target_noise, reduction="none")

        # sigmoid loss weighting
        # proposed by Kingma & Gao (2023, https://arxiv.org/abs/2303.00848)
        # further studied in Simple Diffusion 2 (2024, https://arxiv.org/abs/2410.19324)
        loss_weight = torch.sigmoid(self.sigmoid_bias - logsnr)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss

    def ddim_sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
    ):
        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)
        # print("curr_noise_level", curr_noise_level)

        alpha = self.alphas_cumprod[clipped_curr_noise_level]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            self.ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        alpha = self.add_shape_channels(alpha)
        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        if guidance_fn is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                model_pred = self.model_predictions(
                    x=x,
                    k=clipped_curr_noise_level,
                    external_cond=external_cond,
                    external_cond_mask=external_cond_mask,
                )

                guidance_loss = guidance_fn(
                    xk=x, pred_x0=model_pred.pred_x_start, alpha_cumprod=alpha
                )

                grad = -torch.autograd.grad(
                    guidance_loss,
                    x,
                )[0]
                grad = torch.nan_to_num(grad, nan=0.0)

                pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * grad
                x_start = torch.where(
                    alpha > 0,  # to avoid NaN from zero terminal SNR
                    self.predict_start_from_noise(
                        x, clipped_curr_noise_level, pred_noise
                    ),
                    model_pred.pred_x_start,
                )

        else:
            model_pred = self.model_predictions(
                x=x,
                k=clipped_curr_noise_level,
                external_cond=external_cond,
                external_cond_mask=external_cond_mask,
            )
            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            x,
            x_pred,
        )

        return x_pred
