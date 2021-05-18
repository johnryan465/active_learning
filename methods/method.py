from abc import ABC, abstractmethod
from enum import Enum

from torch.utils import data
from utils.utils import get_pool

from torchtyping.tensor_type import TensorType
from methods.method_params import MethodParams, UncertainMethodParams

from batchbald_redux.batchbald import CandidateBatch

from models.model import ModelWrapper, UncertainModel
from datasets.activelearningdataset import ActiveLearningDataset, DatasetUtils
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from typing import List
import torch
import torchvision


class MethodName(str, Enum):
    batchbald = 'batchbald'
    bald = 'bald'
    entropy = 'entropy'
    random = 'random'


class Method(ABC):
    def __init__(self, params: MethodParams) -> None:
        super().__init__()
        self.params = params
        self.current_aquisition = 0

    def acquire(self, model: ModelWrapper, dataset: ActiveLearningDataset, tb_logger: TensorboardLogger) -> None:
        """
        Moves data from the pool to training
        """
        with torch.no_grad():
            pool = get_pool(dataset)
            candidate_batch = self.score(model, pool)
        indexes = candidate_batch.indices
        Method.log_batch(dataset.get_indexes(indexes), tb_logger, self.current_aquisition)
        dataset.move(indexes)

        self.current_aquisition += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abstractmethod
    def score(self, model: ModelWrapper, pool: TensorType) -> CandidateBatch:
        """
        Generates the candidate batch
        """
        pass

    def complete(self) -> bool:
        return self.current_aquisition >= self.params.max_num_aquisitions

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        DatasetUtils.balanced_init(dataset, self.params.initial_size, dataset.get_config().smoke_test)

    @staticmethod
    def log_batch(images: List[torch.Tensor], tb_logger: TensorboardLogger, index: int) -> None:
        grid = torchvision.utils.make_grid(images)
        tb_logger.writer.add_image('images', grid, index)


# A type of method which requires a model which can output
# with uncertainty
class UncertainMethod(Method):
    def __init__(self, params: UncertainMethodParams) -> None:
        super().__init__(params)
        self.params = params
    
    @abstractmethod
    def score(self, model: UncertainModel, pool: TensorType) -> CandidateBatch:
        pass
