from dataclasses import dataclass

from batchbald_redux.batchbald import CandidateBatch
from torchtyping.tensor_type import TensorType

from methods.method_params import MethodParams
from .method import Method

from models.model import ModelWrapper
import random


@dataclass
class RandomParams(MethodParams):
    pass


# This method simply aquires in a random order
class Random(Method):
    def __init__(self, params: RandomParams) -> None:
        super().__init__(params)

    def score(self, model: ModelWrapper, inputs: TensorType) -> CandidateBatch:
        idxs = list(random.sample(range(0, inputs.shape[0]), self.params.aquisition_size))
        return CandidateBatch(list([0 for i in range(0, self.params.aquisition_size)]), idxs)

