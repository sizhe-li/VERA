import vera.datasets  # noqa F401
import vera.idm  # noqa F401

from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls


class JacobianLearningExperiment(BaseLightningExperiment):
    compatible_algorithms = {
        "image_jacobian",
        "latent_jacobian",
        "dino_feature_jacobian",
        "dino_chunk_jacobian",
        "dino_feature_jacobian_imageres",
    }
    compatible_datasets = {
        "pusht",
        "iiwa",
        "robomimic",
        "drake_allegro",
        "droid",
        "visualmimicdataset",
    }
    data_module_cls = _data_module_cls
