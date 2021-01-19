from dataclasses import dataclass

from typing import List
from models.model import UncertainModel
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod
from methods.method_params import MethodParams
from batchbald_redux.batchbald import get_bald_batch
import torch


@dataclass
class BALDParams(MethodParams):
    batch_size : int
    samples : int
    max_num_batches : int
    initial_size : int

class BALD(UncertainMethod):
    def __init__(self, params : BALDParams) -> None:
        super().__init__()
        self.params = params
        self.current_batch = 0


    def acquire(self, model: UncertainModel,
                dataset: ActiveLearningDataset) -> None:
        probs = []
        for x, _ in dataset.get_pool():
            x = x.cuda()
            probs_ = model.sample(x, self.params.samples).detach().clone()
            probs.append(probs_)
        
        probs = torch.cat(probs, dim=0)
        idxs = get_bald_batch(probs, self.params.batch_size)
        dataset.move(idxs)
        self.current_batch += 1

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        dataset.move([i for i in range(0, self.params.initial_size )])

    def complete(self) -> bool:
        return self.current_batch >= self.params.max_num_batches
