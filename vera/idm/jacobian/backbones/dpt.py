from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers.models.dpt.modeling_dpt import (
    DPTFeatureFusionLayer,
    DPTModel,
    DPTNeck,
    DPTPreTrainedModel,
    DPTReassembleLayer,
    load_backbone,
)


class ContiguousDPTReassembleLayer(DPTReassembleLayer):
    """Keep neck projection inputs in standard contiguous layout for DDP buckets."""

    def forward(self, hidden_state):
        hidden_state = self.projection(hidden_state.contiguous())
        hidden_state = self.resize(hidden_state.contiguous())
        return hidden_state.contiguous()


class ContiguousDPTFeatureFusionLayer(DPTFeatureFusionLayer):
    """Avoid channels-last grads from resize + 1x1 projection in fusion blocks."""

    def forward(self, hidden_state, residual=None):
        hidden_state = hidden_state.contiguous()
        if residual is not None:
            residual = residual.contiguous()
            if hidden_state.shape != residual.shape:
                residual = F.interpolate(
                    residual,
                    size=(hidden_state.shape[2], hidden_state.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                ).contiguous()
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state.contiguous())
        hidden_state = F.interpolate(
            hidden_state,
            scale_factor=2,
            mode="bilinear",
            align_corners=self.align_corners,
        ).contiguous()
        hidden_state = self.projection(hidden_state)
        return hidden_state.contiguous()


class ContiguousDPTNeck(DPTNeck):
    """Patch the stock DPT neck to preserve a contiguous conv layout."""

    def __init__(self, config):
        super().__init__(config)

        if self.reassemble_stage is not None:
            for idx, layer in enumerate(self.reassemble_stage.layers):
                if not isinstance(layer, DPTReassembleLayer):
                    continue
                patched_layer = ContiguousDPTReassembleLayer(
                    config,
                    channels=config.neck_hidden_sizes[idx],
                    factor=config.reassemble_factors[idx],
                )
                patched_layer.load_state_dict(layer.state_dict())
                self.reassemble_stage.layers[idx] = patched_layer

        for idx, layer in enumerate(self.fusion_stage.layers):
            patched_layer = ContiguousDPTFeatureFusionLayer(
                config,
                align_corners=layer.align_corners,
            )
            patched_layer.load_state_dict(layer.state_dict())
            self.fusion_stage.layers[idx] = patched_layer

    def forward(self, hidden_states: list[torch.Tensor], patch_height=None, patch_width=None) -> list[torch.Tensor]:
        if not isinstance(hidden_states, (tuple, list)):
            raise TypeError("hidden_states should be a tuple or list of tensors")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError(
                "The number of hidden states should be equal to the number of neck hidden sizes."
            )

        if self.reassemble_stage is not None:
            hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [
            self.convs[i](feature.contiguous()).contiguous()
            for i, feature in enumerate(hidden_states)
        ]
        output = self.fusion_stage(features)
        return [feature.contiguous() for feature in output]


class DptWrapper(DPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = None
        if config.is_hybrid is False and (
            config.backbone_config is not None or config.backbone is not None
        ):
            self.backbone = load_backbone(config)
        else:
            self.dpt = DPTModel(config, add_pooling_layer=False)

        # Neck
        self.neck = ContiguousDPTNeck(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> List[torch.Tensor]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, DPTForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        >>> model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # interpolate to original size
        >>> post_processed_output = image_processor.post_process_depth_estimation(
        ...     outputs,
        ...     target_sizes=[(image.height, image.width)],
        ... )

        >>> # visualize the prediction
        >>> predicted_depth = post_processed_output[0]["predicted_depth"]
        >>> depth = predicted_depth * 255 / predicted_depth.max()
        >>> depth = depth.detach().cpu().numpy()
        >>> depth = Image.fromarray(depth.astype("uint8"))
        ```"""
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        if self.backbone is not None:
            outputs = self.backbone.forward_with_filtered_kwargs(
                pixel_values,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            hidden_states = outputs.feature_maps
        else:
            outputs = self.dpt(
                pixel_values,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,  # we need the intermediate hidden states
                return_dict=return_dict,
            )
            hidden_states = outputs.hidden_states if return_dict else outputs[1]
            # only keep certain features based on config.backbone_out_indices
            # note that the hidden_states also include the initial embeddings
            if not self.config.is_hybrid:
                hidden_states = [
                    feature
                    for idx, feature in enumerate(hidden_states[1:])
                    if idx in self.config.backbone_out_indices
                ]

            else:
                backbone_hidden_states = (
                    outputs.intermediate_activations
                    if return_dict
                    else list(outputs[-1])
                )
                backbone_hidden_states.extend(
                    feature
                    for idx, feature in enumerate(hidden_states[1:])
                    if idx in self.config.backbone_out_indices[2:]
                )

                hidden_states = backbone_hidden_states

        patch_height, patch_width = None, None
        if self.config.backbone_config is not None and self.config.is_hybrid is False:
            _, _, height, width = pixel_values.shape
            patch_size = self.config.backbone_config.patch_size
            patch_height = height // patch_size
            patch_width = width // patch_size

        hidden_states = self.neck(hidden_states, patch_height, patch_width)

        return hidden_states
