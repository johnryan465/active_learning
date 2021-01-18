from dataclasses import dataclass
import math
from utils.uncertainty import CandidateBatch, compute_conditional_entropy, compute_entropy

import tqdm
import toma
from typing import List
from models.model import UncertainModel
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod
from methods.method_params import MethodParams

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

    @staticmethod
    def get_bald_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None) -> CandidateBatch:
        N, K, C = log_probs_N_K_C.shape

        batch_size = min(batch_size, N)

        candidate_indices = []

        scores_N = -compute_conditional_entropy(log_probs_N_K_C)
        scores_N += compute_entropy(log_probs_N_K_C)

        candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

        return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist())

    def acquire(self, model: UncertainModel,
                dataset: ActiveLearningDataset) -> None:
        probs = []
        for x, _ in dataset.get_pool():
            x = x.cuda()
            probs_ = model.sample(x, self.params.samples)
            probs.append(probs_)
        
        probs = torch.cat(probs, dim=0)
        batch = BALD.get_bald_batch(probs, self.params.batch_size)
        dataset.move(batch.indices)
        self.current_batch += 1

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        dataset.move([i for i in range(0, self.params.initial_size )])

    def complete(self) -> bool:
        return self.current_batch >= self.params.max_num_batches
