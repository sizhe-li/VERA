import pathlib
from typing import Optional, Union

from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig

from .base_exp import BaseExperiment
from .inverse_dynamics_learning import InverseDynamicsLearningExperiment
from .jacobian_learning import JacobianLearningExperiment
from .joint_motion_jacobian_learning import JointMotionJacobianLearningExperiment
from .motion_policy_learning import MotionPolicyLearningExperiment
from .video_action_latent import VideoActionLatentExperiment
from .video_generation import VideoGenerationExperiment

# each key has to be a yaml file under '[project_root]/configurations/experiment' without .yaml suffix
exp_registry = dict(
    inverse_dynamics_learning=InverseDynamicsLearningExperiment,
    jacobian_learning=JacobianLearningExperiment,
    video_action_latent=VideoActionLatentExperiment,
    motion_policy_learning=MotionPolicyLearningExperiment,
    joint_motion_jacobian_learning=JointMotionJacobianLearningExperiment,
    video_generation=VideoGenerationExperiment,
)


def build_experiment(
    cfg: DictConfig,
    logger: Optional[WandbLogger] = None,
    ckpt_path: Optional[Union[str, pathlib.Path]] = None,
) -> BaseExperiment:
    """
    Build an experiment instance based on registry
    :param cfg: configuration file
    :param logger: optional logger for the experiment
    :param ckpt_path: optional checkpoint path for saving and loading
    :return:
    """
    if cfg.experiment._name not in exp_registry:
        raise ValueError(
            f"Experiment {cfg.experiment._name} not found in registry {list(exp_registry.keys())}. "
            "Make sure you register it correctly in 'experiments/__init__.py' under the same name as yaml file."
        )

    return exp_registry[cfg.experiment._name](cfg, logger, ckpt_path)
