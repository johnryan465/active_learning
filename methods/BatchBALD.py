from datasets.activelearningdataset import DatasetUtils
from models.model import UncertainModel
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod
from methods.method_params import MethodParams
from batchbald_redux.batchbald import get_batchbald_batch


import torch
from tqdm import tqdm
from typing import List
from dataclasses import dataclass


@dataclass
class BatchBALDParams(MethodParams):
    batch_size : int
    samples : int
    max_num_batches : int
    initial_size : int
    use_cuda : bool


class BatchBALD(UncertainMethod):
    def __init__(self, params : BatchBALDParams) -> None:
        super().__init__()
        self.params = params
        self.current_batch = 0

    def acquire(self, model: UncertainModel, dataset: ActiveLearningDataset) -> None:
        probs = []
        for x, _ in tqdm(dataset.get_pool(), desc="Sampling", leave=False):
            if self.params.use_cuda:
                x = x.cuda()
            probs_ = model.sample(x, self.params.samples).detach().clone()
            probs.append(probs_)
        
        probs = torch.cat(probs, dim=0)
        print(probs.shape)
        batch = get_batchbald_batch(probs, self.params.batch_size, self.params.samples)
        dataset.move(batch.indices)
        self.current_batch += 1

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        DatasetUtils.balanced_init(dataset, self.params.initial_size)

    def complete(self) -> bool:
        return self.current_batch >= self.params.max_num_batches