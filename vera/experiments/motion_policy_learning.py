import vera.datasets  # noqa F401
import vera.idm  # noqa F401

from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls


class MotionPolicyLearningExperiment(BaseLightningExperiment):

    compatible_algorithms = {"dfot_motion_policy", "dfot_motion_policy_joint"}
    compatible_datasets = {"pusht", "iiwa", "robomimic"}
    data_module_cls = _data_module_cls
