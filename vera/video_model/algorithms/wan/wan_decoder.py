import torch
import torch.nn as nn
from einops import rearrange
from lightning.pytorch.utilities import grad_norm
from omegaconf import OmegaConf
from pathlib import Path
from transformers import get_scheduler

from vera.video_model.algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .modules.vae import video_vae_factory
from .utils.optical_flow import flow_to_rgb
from vera.video_model.utils.distributed_utils import is_rank_zero


class WanDecoder(BasePytorchAlgo):
    """
    Train the Wan VAE decoder only.
    """

    def __init__(self, cfg):
        self.load_video_latent = cfg.load_video_latent
        super().__init__(cfg)

    @staticmethod
    def classes_to_shard():
        return set()

    def configure_model(self):
        self.train_head_only_steps = int(getattr(self.cfg, "train_head_only_steps", 0))
        self.non_head_lr_scale = float(getattr(self.cfg, "non_head_lr_scale", 1.0))
        self._non_head_frozen = False

        out_channels = getattr(self.cfg.vae, "out_channels", 2)
        self.vae = video_vae_factory(
            pretrained_path=self.cfg.vae.ckpt_path,
            z_dim=self.cfg.vae.z_dim,
            out_channels=out_channels,
        ).train()
        self.decoder_head = self.vae.decoder.head[2]
        decoder_dtype = next(self.vae.decoder.parameters()).dtype
        self.decoder_head.to(dtype=decoder_dtype)
        self.vae.requires_grad_(False)
        self.vae.decoder.requires_grad_(True)
        self.vae.conv2.requires_grad_(True)
        self.decoder_head.requires_grad_(True)

        # freeze encoder on flow training
        self.vae.encoder.requires_grad_(False).eval()
        self.vae.conv1.requires_grad_(False).eval()
        if self.train_head_only_steps > 0:
            self.vae.decoder.requires_grad_(False)
            self.vae.conv2.requires_grad_(False)
            self.decoder_head.requires_grad_(True)
            self._non_head_frozen = True

        self.register_buffer("vae_mean", torch.tensor(self.cfg.vae.mean))
        self.register_buffer("vae_inv_std", 1.0 / torch.tensor(self.cfg.vae.std))
        # NOTE: do NOT cache [self.vae_mean, self.vae_inv_std] in a plain list.
        # register_buffer tensors are replaced by .to(device); a cached list keeps stale CPU refs.

        # pseudo-huber loss (robust L1) with no reduction
        self.criterion = lambda pred, target: torch.sqrt((pred - target) ** 2 + 1e-6)

    def configure_optimizers(self):
        head_params = list(self.decoder_head.parameters())
        head_param_ids = {id(p) for p in head_params}
        non_head_params = [
            p
            for p in list(self.vae.decoder.parameters()) + list(self.vae.conv2.parameters())
            if id(p) not in head_param_ids
        ]

        param_groups = [{"params": head_params, "lr": self.cfg.lr}]
        if non_head_params:
            param_groups.append(
                {"params": non_head_params, "lr": self.cfg.lr * self.non_head_lr_scale}
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
        )
        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer,
                **self.cfg.lr_scheduler,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def on_train_batch_start(self, batch, batch_idx):
        if self._non_head_frozen and self.global_step >= self.train_head_only_steps:
            self.vae.decoder.requires_grad_(True)
            self.vae.conv2.requires_grad_(True)
            self._non_head_frozen = False
        return super().on_train_batch_start(batch, batch_idx)

    @torch.no_grad()
    def on_after_batch_transfer(self, batch, dataloader_idx=0):
        return batch

    def decode(self, latents):
        vae_output = self.vae.decode(latents, [self.vae_mean, self.vae_inv_std])
        sliced_output = vae_output[:, :, 1:, :, :]
        return sliced_output

    def compute_flow_loss(self, pred_flows, gt_flows, flow_valid_mask=None):
        b, c, t, h, w = pred_flows.shape
        se = self.criterion(pred_flows, gt_flows)

        if flow_valid_mask is None:
            return se.mean()

        assert t == flow_valid_mask.shape[1], "pred and gt frame mismatch"

        mask = flow_valid_mask.to(device=se.device, dtype=se.dtype)
        mask = rearrange(mask, "b t -> b 1 t 1 1")
        masked_se = se * mask
        denom = mask.sum().clamp_min(1.0) * c * h * w
        return masked_se.sum() / denom

    def training_step(self, batch, batch_idx, *args, **kwargs):
        batch_size = batch["videos"].shape[0]
        if self.load_video_latent:
            video_latents = batch["video_latents"]
        else:
            videos = rearrange(batch["videos"], "b t c h w -> b c t h w")
            with torch.no_grad():
                video_latents, _ = self.vae.encode_mu_var(videos, [self.vae_mean, self.vae_inv_std])

        pred_flows = self.decode(video_latents)
        gt_flows = rearrange(batch["optical_flow"], "b t c h w -> b c t h w")
        flow_valid_mask = batch.get("optical_flow_valid_mask")
        loss = self.compute_flow_loss(pred_flows, gt_flows, flow_valid_mask)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Use Lightning utility to compute parameter gradient norms.
        norms = grad_norm(self, norm_type=2)
        total_key = next((k for k in norms if k.endswith("_total")), None)
        if total_key is not None:
            grad_total = norms[total_key]
            if isinstance(grad_total, torch.Tensor):
                grad_total = grad_total.to(self.device)
            self.log(
                "train/grad_norm",
                grad_total,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

        # Log LR curves for each optimizer param group.
        for group_idx, param_group in enumerate(optimizer.param_groups):
            lr_name = "train/lr" if group_idx == 0 else f"train/lr_group_{group_idx}"
            self.log(
                lr_name,
                float(param_group["lr"]),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

    @torch.no_grad()
    def visualize(self, videos, pred_flows, gt_flows, mask=None):
        if is_rank_zero:
            rgb = rearrange(videos[:, 1:], "b t c h w -> b c t h w")
            rgb = rgb.mul(0.5).add(0.5).clamp(0, 1)
            gt_rgb_flow = flow_to_rgb(gt_flows)
            pred_rgb_flow = flow_to_rgb(pred_flows)

            if mask is not None:
                mask = rearrange(mask, "b t -> b 1 t 1 1")
                gt_rgb_flow = gt_rgb_flow * mask
                pred_rgb_flow = pred_rgb_flow * mask
                rgb = rgb * mask

            video_vis = torch.cat([rgb, gt_rgb_flow, pred_rgb_flow], dim=-1)
            video_vis = rearrange(video_vis, "b c t h w -> b t c h w").cpu()

            self.log_video(
                "val_vis/rgb_flow_predflow",
                video_vis,
                fps=self.cfg.logging.fps,
                step=self.global_step,
            )

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        batch_size = batch["videos"].shape[0]
        if self.load_video_latent:
            video_latents = batch["video_latents"]
        else:
            videos = rearrange(batch["videos"], "b t c h w -> b c t h w")
            with torch.no_grad():
                video_latents, _ = self.vae.encode_mu_var(videos, [self.vae_mean, self.vae_inv_std])

        pred_flows = self.decode(video_latents)
        gt_flows = rearrange(batch["optical_flow"], "b t c h w -> b c t h w")
        flow_valid_mask = batch.get("optical_flow_valid_mask")
        loss = self.compute_flow_loss(pred_flows, gt_flows, flow_valid_mask)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)

        self.visualize(batch["videos"], pred_flows, gt_flows, flow_valid_mask)
        return loss


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "configurations/algorithm/wan_decoder.yaml"
    cfg = OmegaConf.load(cfg_path)
    vae_cfg = cfg.vae

    ckpt_path = repo_root / vae_cfg.ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_path}")

    decoder = WanDecoder.__new__(WanDecoder)
    nn.Module.__init__(decoder)
    decoder.vae = video_vae_factory(
        pretrained_path=str(ckpt_path),
        z_dim=vae_cfg.z_dim,
        out_channels=vae_cfg.out_channels,
    ).eval()
    decoder.register_buffer("vae_mean", torch.tensor(vae_cfg.mean))
    decoder.register_buffer("vae_inv_std", 1.0 / torch.tensor(vae_cfg.std))
    decoder.criterion = nn.MSELoss(reduction='none')
    decoder.load_video_latent = False
    decoder.log = lambda *args, **kwargs: None

    batch_size, num_frames, channels, height, width = 2, 9, 3, 16, 16
    videos = torch.randn(batch_size, num_frames, channels, height, width)
    optical_flow = torch.randn(batch_size, num_frames - 1, 2, height, width)
    optical_flow_valid_mask = torch.ones(batch_size, num_frames - 1, dtype=torch.bool)
    batch = {
        "videos": videos,
        "optical_flow": optical_flow,
        "optical_flow_valid_mask": optical_flow_valid_mask,
    }

    batch = decoder.on_after_batch_transfer(batch, dataloader_idx=0)
    loss = decoder.training_step(batch, batch_idx=0)
    print(f"[wan_decoder smoke test] train/loss={loss.item():.6f}")
