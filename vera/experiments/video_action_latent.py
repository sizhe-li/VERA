import vera.datasets  # noqa F401
import vera.idm  # noqa F401

from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls


class VideoActionLatentExperiment(BaseLightningExperiment):

    compatible_algorithms = {"dfot_video_action_latent"}
    compatible_datasets = {"pusht", "iiwa"}

    data_module_cls = _data_module_cls
