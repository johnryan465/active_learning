from abc import ABC, abstractmethod
from enum import Enum

from models.model import ModelWrapper, UncertainModel
from datasets.activelearningdataset import ActiveLearningDataset
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from typing import List
import torch
import torchvision


class MethodName(str, Enum):
    batchbald = 'batchbald'
    bald = 'bald'
    random = 'random'


class Method(ABC):
    @abstractmethod
    def acquire(self, model: ModelWrapper, dataset: ActiveLearningDataset, tb_logger: TensorboardLogger) -> None:
        """
        Moves data from the pool to training
        """
        pass

    @abstractmethod
    def complete(self) -> bool:
        """
        Checks whether or not the method is complete
        """
        pass

    @abstractmethod
    def initialise(self, dataset: ActiveLearningDataset) -> None:
        """
        Initialise the dataset to have some training data
        """
        pass

    @staticmethod
    def log_batch(images: List[torch.Tensor], tb_logger: TensorboardLogger, index: int) -> None:
        grid = torchvision.utils.make_grid(images)
        tb_logger.writer.add_image('images', grid, index)


# A type of method which requires a model which can output
# with uncertainty
class UncertainMethod(Method):
    @abstractmethod
    def acquire(self, model: UncertainModel, dataset: ActiveLearningDataset) -> None:
        pass
