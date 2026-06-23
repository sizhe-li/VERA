import torch
import torch.nn as nn

from einops import rearrange
from diffusers import AutoencoderKLCogVideoX
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from vera.video_model.algorithms.common.base_pytorch_algo import BasePytorchAlgo


class CogVideoXVAE(BasePytorchAlgo):
    """
    Main classs for CogVideoXImageToVideo
    """

    def __init__(self, cfg):
        self.pretrained_cfg = cfg.pretrained
        super().__init__(cfg)

    def configure_model(self):
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            self.pretrained_cfg.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.pretrained_cfg.revision,
            variant=self.pretrained_cfg.variant,
        )
        self.criteria = nn.MSELoss()

    @torch.no_grad()
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # data reprocessing, returned result is passed to self.training_step / self.validation_step

        images = batch["images"]
        videos = batch["videos"]
        batch_size = images.size(0)

        # Encode videos
        if not self.cfg.load_video_latent:
            images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
            image_noise_sigma = torch.normal(
                mean=-3.0,
                std=0.5,
                size=(batch_size,),
                device=self.device,
                dtype=images.dtype,
            )
            image_noise_sigma = torch.exp(image_noise_sigma)
            noisy_images = (
                images
                + torch.randn_like(images)
                * image_noise_sigma[:, None, None, None, None]
            )
            if self.trainer.training:
                image_latent_dist = self.vae.encode(noisy_images).latent_dist
            else:
                image_latent_dist = self.vae.encode(images).latent_dist
            videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
            latent_dist = self.vae.encode(videos).latent_dist
        else:
            image_latent_dist = DiagonalGaussianDistribution(images)
            latent_dist = DiagonalGaussianDistribution(videos)
        if self.trainer.training:
            image_latents = image_latent_dist.mode().clone()
            video_latents = latent_dist.mode().clone()
        else:
            image_latents = image_latent_dist.sample()
            video_latents = latent_dist.sample()

        batch["image_latents"] = image_latents
        batch["video_latents"] = video_latents

        return batch

    def training_step(self, batch, batch_idx, *args, **kwargs):
        image_latents = batch["image_latents"]
        video_latents = batch["video_latents"]

        # video_pred = self.vae.decode(video_latents, return_dict=False)
        # loss = self.criteria(video_pred, batch["videos"])
        raise NotImplementedError(
            "VAE is inference only. Append experiment.tasks=[validation] to run inference"
        )

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        image_latents = batch["image_latents"]
        video_latents = batch["video_latents"]

        video_pred, *_ = self.vae.decode(video_latents, return_dict=False)
        video_pred = rearrange(video_pred, "b c t h w -> b t c h w")
        video_gt = batch["videos"]
        video = torch.cat([video_gt, video_pred], dim=-1)
        video = video * 0.5 + 0.5
        video = video.cpu()
        video = rearrange(self.all_gather(video), "p b ... -> (p b) ...")
        self.log_video("validation_vis/video_pred", video)

        return video_pred
