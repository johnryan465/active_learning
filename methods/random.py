from marshmallow_dataclass import dataclass

from params.method_params import MethodParams
from .method import Method

from models.model import ModelWrapper
from datasets.activelearningdataset import ActiveLearningDataset


@dataclass
class RandomParams(MethodParams):
    batch_size : int

class Random(Method):
    def __init__(self, batch_size: int, max_num_batches: int,
                 initial_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.initial_size = initial_size
        self.max_num_batches = max_num_batches
        self.current_batch = 0
        self.last_index = 0

    def acquire(self, model: ModelWrapper, dataset: ActiveLearningDataset) -> None:
        dataset.move(
            [i + self.last_index
                for i in range(0, self.batch_size)]
        )
        self.current_batch = self.current_batch + 1
        self.last_index = self.last_index + (
            self.current_batch * self.batch_size)

    def complete(self) -> bool:
        return self.current_batch >= self.max_num_batches

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        dataset.move([i for i in range(0, self.initial_size)])
        self.last_index = self.initial_size
