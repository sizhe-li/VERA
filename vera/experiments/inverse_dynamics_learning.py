import vera.idm  # noqa: F401
import vera.datasets  # noqa: F401

from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls


class InverseDynamicsLearningExperiment(BaseLightningExperiment):
    compatible_algorithms = {
        "pusht_inverse_diffusion",
        "pure_transformer_action",
        "pure_transformer_dino_action",
        "dpt_vggt_pooled_action",
    }
    compatible_datasets = {"pusht", "drake_allegro", "robomimic", "iiwa"}
    data_module_cls = _data_module_cls
