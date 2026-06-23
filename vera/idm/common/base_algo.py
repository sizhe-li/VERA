from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BaseAlgoCfg:
    """
    A base configuration dataclass for algorithms.
    """

    debug: bool = False


class BaseAlgo(ABC):
    """
    A base class for generic algorithms.
    """

    def __init__(self, cfg: BaseAlgoCfg):
        super().__init__()
        self.cfg = cfg
        self.debug = self.cfg.debug

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the algorithm.
        """
        raise NotImplementedError
