from typing import Sequence
import torch

from transformers import AutoTokenizer, T5EncoderModel

from vera.video_model.algorithms.common.base_pytorch_algo import BasePytorchAlgo
from vera.video_model.algorithms.cogvideo.text_encoder import compute_prompt_embeddings


class T5Encoder(BasePytorchAlgo):
    def __init__(self, cfg):
        self.pretrained_cfg = cfg.pretrained
        self.max_text_seq_length = 226
        super().__init__(cfg)
        self._build_model()  # Explicitly call _build_model after initialization

    def _build_model(self):
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

    def training_step(self, *args, **kwargs):
        raise NotImplementedError("T5Encoder does not support training")

    @torch.no_grad()
    def predict(self, prompts: Sequence[str]):
        prompt_embeds = compute_prompt_embeddings(
            self.tokenizer,
            self.text_encoder,
            prompts,
            self.max_text_seq_length,
            self.device,
            torch.bfloat16,
            requires_grad=False,
        )
        return prompt_embeds
