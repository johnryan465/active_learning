from marshmallow_dataclass import dataclass

from methods.method_params import MethodParams
from .method import Method

from models.model import ModelWrapper
from datasets.activelearningdataset import ActiveLearningDataset, DatasetUtils
import random


@dataclass
class RandomParams(MethodParams):
    pass


# This method simply aquires in a random order
class Random(Method):
    def __init__(self, params: RandomParams) -> None:
        super().__init__()
        self.params = params
        self.current_aquisition = 0

    def acquire(self, model: ModelWrapper, dataset: ActiveLearningDataset) -> None:
        dataset.move(list(random.sample(range(0, dataset.get_pool_size()), self.params.aquisition_size)))
        self.current_aquisition = self.current_aquisition + 1

    def complete(self) -> bool:
        return self.current_aquisition >= self.params.max_num_aquisitions

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        DatasetUtils.balanced_init(dataset, self.params.initial_size)
