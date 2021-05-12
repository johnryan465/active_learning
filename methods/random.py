from dataclasses import dataclass

from batchbald_redux.batchbald import CandidateBatch

from methods.method_params import MethodParams
from .method import Method

from models.model import ModelWrapper
from datasets.activelearningdataset import ActiveLearningDataset, DatasetUtils
import random
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger


@dataclass
class RandomParams(MethodParams):
    pass


# This method simply aquires in a random order
class Random(Method):
    def __init__(self, params: RandomParams) -> None:
        super().__init__(params)

    def acquire(self, model: ModelWrapper, dataset: ActiveLearningDataset, tb_logger: TensorboardLogger) -> CandidateBatch:
        idxs = list(random.sample(range(0, dataset.get_pool_size()), self.params.aquisition_size))
        return CandidateBatch(list([0 for i in range(0, self.params.aquisition_size)]), idxs)

