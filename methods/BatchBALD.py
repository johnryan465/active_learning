from methods.BALD import CandidateBatch, compute_conditional_entropy
from models.model import UncertainModel
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod
from methods.method_params import MethodParams
import uncertainty.joint_entropy as joint_entropy
import torch


import torch
from toma import toma
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

    def get_batchbald_batch(
            log_probs_N_K_C: torch.Tensor, batch_size: int, num_samples: int, dtype=None, device=None
        ) -> CandidateBatch:
        N, K, C = log_probs_N_K_C.shape

        batch_size = min(batch_size, N)

        candidate_indices = []
        candidate_scores = []

        if batch_size == 0:
            return CandidateBatch(candidate_scores, candidate_indices)

        conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)

        batch_joint_entropy = joint_entropy.DynamicJointEntropy(
            num_samples, batch_size - 1, K, C, dtype=dtype, device=device
        )

        # We always keep these on the CPU.
        scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

        for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
            if i > 0:
                latest_index = candidate_indices[-1]
                batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

            shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

            batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

            scores_N -= conditional_entropies_N + shared_conditinal_entropies
            scores_N[candidate_indices] = -float("inf")

            candidate_score, candidate_index = scores_N.max(dim=0)

            candidate_indices.append(candidate_index.item())
            candidate_scores.append(candidate_score.item())

        return CandidateBatch(candidate_scores, candidate_indices)

    def acquire(self, model: UncertainModel,
                dataset: ActiveLearningDataset) -> None:
        probs = []
        for x, _ in dataset.get_pool():
            x = x.cuda()
            probs_ = model.sample(x, self.params.samples)
            probs.append(probs_)
        
        probs = torch.cat(probs, dim=0)
        batch = BatchBALD.get_batchbald_batch(probs, self.params.batch_size, self.params.samples)
        dataset.move(batch.indices)
        self.current_batch += 1

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        dataset.move([i for i in range(0, self.params.initial_size )])

    def complete(self) -> bool:
        return self.current_batch >= self.params.max_num_batches