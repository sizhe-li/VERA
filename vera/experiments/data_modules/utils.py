from omegaconf import DictConfig

# from vera.datasets.video.base_video import (
#     BaseAdvancedVideoDataset,
# )
from . import BaseDataModule


def _data_module_cls(_, root_cfg: DictConfig, compatible_datasets: set[str]):
    # dataset_cls = compatible_datasets[root_cfg.dataset._name]
    # data_module_cls = None
    # if issubclass(dataset_cls, BaseAdvancedVideoDataset):
    #     data_module_cls = ResumableDataModule
    # else:
    data_module_cls = BaseDataModule
    return data_module_cls(root_cfg, compatible_datasets)
