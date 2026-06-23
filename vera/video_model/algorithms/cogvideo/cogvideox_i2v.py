import torch
import random
import numpy as np
from einops import rearrange

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import AutoTokenizer, T5EncoderModel

# from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from vera.video_model.algorithms.common.base_pytorch_algo import BasePytorchAlgo
from vera.video_model.algorithms.cogvideo.text_encoder import compute_prompt_embeddings
from vera.video_model.algorithms.cogvideo.pos_embed import prepare_rotary_positional_embeddings
from vera.video_model.utils.distributed_utils import is_rank_zero


class CogVideoXImageToVideo(BasePytorchAlgo):
    """
    Main classs for CogVideoXImageToVideo
    """

    def __init__(self, cfg):
        self.pretrained_cfg = cfg.pretrained
        super().__init__(cfg)

    @staticmethod
    def classes_to_shard():
        classes = {CogVideoXBlock}
        return classes

    def configure_model(self):
        if not self.cfg.load_prompt_embed:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_cfg.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.pretrained_cfg.revision,
            )

            self.text_encoder = T5EncoderModel.from_pretrained(
                self.pretrained_cfg.pretrained_model_name_or_path,
                subfolder="text_encoder",
                revision=self.pretrained_cfg.revision,
            )
            self.text_encoder.requires_grad_(False)

        transformer = CogVideoXTransformer3DModel.from_pretrained(
            self.pretrained_cfg.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            revision=self.pretrained_cfg.revision,
            variant=self.pretrained_cfg.variant,
        ).train()
        transformer.enable_gradient_checkpointing()

        # hack to support multi resolution
        if hasattr(transformer.patch_embed, "pos_embedding"):
            del transformer.patch_embed.pos_embedding
            transformer.patch_embed.use_learned_positional_embeddings = False
            transformer.config.use_learned_positional_embeddings = False
        self.transformer = transformer

        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            self.pretrained_cfg.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.pretrained_cfg.revision,
            variant=self.pretrained_cfg.variant,
        )
        self.vae.requires_grad_(False)

        self.scheduler = CogVideoXDPMScheduler.from_pretrained(
            self.pretrained_cfg.pretrained_model_name_or_path, subfolder="scheduler"
        )

        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_cfg.pretrained_model_name_or_path,
            # tokenizer=self.tokenizer,
            text_encoder=self.text_encoder if not self.cfg.load_prompt_embed else None,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.scheduler,
            torch_dtype=torch.bfloat16,
            revision=self.pretrained_cfg.revision,
            variant=self.pretrained_cfg.variant,
        )
        self.pipe = pipe

        # alias parameters
        self.vae_scaling_factor = self.vae.config.scaling_factor
        self.vae_scale_factor_spatial = 2 ** (
            len(self.vae.config.block_out_channels) - 1
        )
        transformer_cfg = (
            self.transformer.module.config
            if hasattr(self.transformer, "module")
            else self.transformer.config
        )
        if "1.5" in self.pretrained_cfg.pretrained_model_name_or_path:
            transformer_cfg.sample_height = 768 // 2
            transformer_cfg.sample_width = 768 // 2
        self.rope_base_height = transformer_cfg.sample_height * self.vae_scaling_factor
        self.rope_base_width = transformer_cfg.sample_width * self.vae_scaling_factor

        self.patch_size = transformer_cfg.patch_size
        self.patch_size_t = (
            transformer_cfg.patch_size_t
            if hasattr(transformer_cfg, "patch_size_t")
            else None
        )
        self.ofs_embed_dim = (
            transformer_cfg.ofs_embed_dim
            if hasattr(transformer_cfg, "ofs_embed_dim")
            else None
        )
        self.attention_head_dim = transformer_cfg.attention_head_dim
        self.use_rotary_positional_embeddings = (
            transformer_cfg.use_rotary_positional_embeddings
        )
        self.max_text_seq_length = transformer_cfg.max_text_seq_length

    def configure_optimizers(self):
        transformer_parameters = list(
            filter(lambda p: p.requires_grad, self.transformer.parameters())
        )
        optimizer = torch.optim.AdamW(
            transformer_parameters,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
        )
        return optimizer

    def on_after_batch_transfer(self, batch, dataloader_idx):
        images = batch["images"]
        videos = batch["videos"]
        prompts = batch["prompts"]

        batch_size = images.size(0)

        # Encode videos
        if not self.cfg.load_video_latent:
            images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
            image_noise_sigma = torch.normal(
                mean=-3.0, std=0.5, size=(batch_size,), device=self.device
            )
            image_noise_sigma = torch.exp(image_noise_sigma)
            noisy_images = images
            if self.trainer.training:
                noisy_images += (
                    torch.randn_like(images)
                    * image_noise_sigma[:, None, None, None, None]
                )
            image_latent_dist = self.vae.encode(noisy_images).latent_dist
            videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
            latent_dist = self.vae.encode(videos).latent_dist
        else:
            image_latent_dist = DiagonalGaussianDistribution(images)
            latent_dist = DiagonalGaussianDistribution(videos)
        if self.trainer.training:
            image_latents = image_latent_dist.mean.clone()
            video_latents = latent_dist.mean.clone()
        else:
            image_latents = image_latent_dist.sample()
            video_latents = latent_dist.sample()

        image_latents = image_latents * self.vae_scaling_factor
        image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        image_latents = image_latents.contiguous()

        video_latents = video_latents * self.vae_scaling_factor
        video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        video_latents = video_latents.contiguous()

        padding_shape = (
            video_latents.shape[0],
            video_latents.shape[1] - 1,
            *video_latents.shape[2:],
        )
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        if random.random() < self.cfg.noised_image_dropout:
            image_latents = torch.zeros_like(image_latents)

        # Encode prompts
        if not self.cfg.load_prompt_embed:
            prompt_embeds = compute_prompt_embeddings(
                self.tokenizer,
                self.text_encoder,
                prompts,
                self.max_text_seq_length,
                self.device,
                torch.bfloat16,
                requires_grad=False,
            )
        else:
            prompt_embeds = batch["prompt_embeds"].to(dtype=torch.bfloat16)
        batch["image_latents"] = image_latents
        batch["video_latents"] = video_latents
        batch["prompt_embeds"] = prompt_embeds

        return batch

    def training_step(self, batch, batch_idx, *args, **kwargs):
        # fill in training step logic here
        image_latents = batch["image_latents"]
        latent_gt = batch["video_latents"]
        prompt_embeds = batch["prompt_embeds"]

        # Sample noise that will be added to the latents
        noise = torch.randn_like(latent_gt)
        batch_size, num_frames, _, height, width = latent_gt.shape

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        )

        # Prepare rotary embeds
        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=height * self.vae_scale_factor_spatial,
                width=width * self.vae_scale_factor_spatial,
                num_frames=num_frames,
                vae_scale_factor_spatial=self.vae_scale_factor_spatial,
                patch_size=self.patch_size,
                patch_size_t=self.patch_size_t,
                attention_head_dim=self.attention_head_dim,
                base_height=self.rope_base_height,
                base_width=self.rope_base_width,
                device=self.device,
            )
            if self.use_rotary_positional_embeddings
            else None
        )

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_video_latents = self.scheduler.add_noise(latent_gt, noise, timesteps)
        noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)

        ofs_emb = (
            None
            if self.ofs_embed_dim is None
            else noisy_model_input.new_full((1,), fill_value=2.0)
        )

        # Predict the noise residual
        model_output = self.transformer(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        latent_pred = self.scheduler.get_velocity(
            model_output, noisy_video_latents, timesteps
        )

        alphas_cumprod = self.scheduler.alphas_cumprod.to(
            self.device, dtype=torch.float32
        )
        weights = 1 / (1 - alphas_cumprod[timesteps])
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = (weights * (latent_pred - latent_gt) ** 2).mean()

        if self.global_step % self.cfg.logging.loss_freq == 0:
            self.log("train/loss", loss, sync_dist=True)

        if self.global_step % self.cfg.logging.video_freq == 0:
            latent_pred = rearrange(latent_pred, "b t c h w -> b c t h w")
            video_pred, *_ = self.vae.decode(latent_pred, return_dict=False)
            video_pred = rearrange(video_pred, "b c t h w -> b t c h w")
            video_gt = batch["videos"]
            video = torch.cat([video_pred, video_gt], dim=-1).cpu() * 0.5 + 0.5
            video = rearrange(self.all_gather(video), "p b ... -> (p b) ...")
            if is_rank_zero:
                self.log_video(
                    "training_vis/video_pred", video[:8], fps=self.cfg.logging.fps
                )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        if self.cfg.load_prompt_embed and self.cfg.sampling.guidance_scale > 1.0:
            raise ValueError(
                "When trying a save training memory with cfg.load_prompt_embed=True, "
                "validation guidance_scale must be set to to 1.0. You can set "
                "guidance_scale to any value after training is done, by setting "
                "cfg.load_prompt_embed=False."
            )

        images = batch["images"]
        prompt_embeds = batch["prompt_embeds"]
        video_gt = batch["videos"]

        video_pred = self.pipe(
            image=torch.clamp(images[:, 0], -1, 1) * 0.5 + 0.5,
            prompt_embeds=prompt_embeds,
            guidance_scale=self.cfg.sampling.guidance_scale,
            use_dynamic_cfg=self.cfg.sampling.use_dynamic_cfg,
            height=images.shape[-2],
            width=images.shape[-1],
            num_frames=video_gt.shape[1],
            output_type="np",
        ).frames
        video_pred = torch.from_numpy(video_pred)
        video_pred = rearrange(video_pred, "b t h w c-> b t c h w")
        video_gt = video_gt.cpu() * 0.5 + 0.5
        video = torch.cat([video_pred, video_gt], axis=-1)
        video = rearrange(self.all_gather(video), "p b ... -> (p b) ...")
        if is_rank_zero:
            self.log_video(
                "validation_vis/video_pred", video[:8], fps=self.cfg.logging.fps
            )
