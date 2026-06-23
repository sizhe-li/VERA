import vera.datasets  # noqa F401
import vera.idm  # noqa F401

from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls


class JointMotionJacobianLearningExperiment(BaseLightningExperiment):
    """
    Experiment wrapper for joint DFoT motion + Jacobian learning.

    This mirrors `MotionPolicyLearningExperiment` and `JacobianLearningExperiment`
    but is wired to the new `dfot_motion_jacobian_joint` algorithm.
    """

    compatible_algorithms = {"dfot_motion_jacobian_joint"}
    compatible_datasets = {"pusht", "iiwa", "robomimic"}
    data_module_cls = _data_module_cls
