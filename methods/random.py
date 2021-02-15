from marshmallow_dataclass import dataclass

from methods.method_params import MethodParams
from .method import Method

from models.model import ModelWrapper
from datasets.activelearningdataset import ActiveLearningDataset
import random


@dataclass
class RandomParams(MethodParams):
    batch_size: int
    max_num_batches: int
    initial_size: int


# This method simply aquires in a random order
class Random(Method):
    def __init__(self, params: RandomParams) -> None:
        super().__init__()
        self.batch_size = params.batch_size
        self.initial_size = params.initial_size
        self.max_num_batches = params.max_num_batches
        self.current_batch = 0

    def acquire(self, model: ModelWrapper, dataset: ActiveLearningDataset) -> None:
        dataset.move(list(random.sample(range(0, dataset.get_pool_size()), self.batch_size)))
        self.current_batch = self.current_batch + 1

    def complete(self) -> bool:
        return self.current_batch >= self.max_num_batches

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        dataset.move([i for i in range(0, self.initial_size)])
