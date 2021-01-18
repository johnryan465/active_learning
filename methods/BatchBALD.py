from models.model import UncertainModel
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod

import torch


import torch
from toma import toma
from tqdm import tqdm
import math
from typing import List
from dataclasses import dataclass


@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(
            log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N

class BatchBALD(UncertainMethod):
    def __init__(self) -> None:
        super().__init__()
        self.num_inference_samples = 5
        self.num_classes = 10
        self.use_cuda = True

    def get_batchbald_batch(
            log_probs_N_K_C: torch.Tensor, batch_size: int, num_samples: int, dtype=None, device=None) -> CandidateBatch:
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
        scores_N = torch.empty(N, dtype=torch.double,
                            pin_memory=torch.cuda.is_available())

        for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
            if i > 0:
                latest_index = candidate_indices[-1]
                batch_joint_entropy.add_variables(
                    log_probs_N_K_C[latest_index: latest_index + 1])

            shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum(
            )

            batch_joint_entropy.compute_batch(
                log_probs_N_K_C, output_entropies_B=scores_N)

            scores_N -= conditional_entropies_N + shared_conditinal_entropies
            scores_N[candidate_indices] = -float("inf")

            candidate_score, candidate_index = scores_N.max(dim=0)

            candidate_indices.append(candidate_index.item())
            candidate_scores.append(candidate_score.item())

        return CandidateBatch(candidate_scores, candidate_indices)

    def acquire(self, model: UncertainModel,
                dataset: ActiveLearningDataset) -> None:
        # Acquire pool predictions
        N = len(dataset.get_pool())
        logits_N_K_C = torch.empty((N, self.num_inference_samples *
                                    self.num_inference_samples,
                                    self.num_classes),
                                   dtype=torch.double,
                                   pin_memory=self.use_cuda)

        with torch.no_grad():
            model.eval()

            for i, (data, _) in enumerate(
                    tqdm(pool_loader,
                         desc="Evaluating Acquisition Set",
                         leave=False)):
                data = data.to(device=device)
                # print(data.shape)

                lower = i * pool_loader.batch_size
                upper = min(lower + pool_loader.batch_size, N)
                b_s = upper-lower
                mc_outputs = model.feature_extractor.random_sample
        (data, num_inference_samples).transpose(0, 1)
        # print(mc_outputs.shape)
        mc_outputs = torch.reshape(mc_outputs, (b_s *
                                                num_inference_samples, 144))
        outputs = model.gp(mc_outputs)
        # print(outputs.shape)
        # print(mc_outputs.shape)
        #ss = torch.Size((num_inference_samples,))
        # MC sampls
        samples = outputs.sample_n(num_inference_samples)
        # print(samples.shape)
        samples = torch.reshape(samples,
                                (num_inference_samples, b_s, num_inference_samples, 10))
        samples = samples.transpose(0, 1)
        samples = torch.reshape(samples, (b_s, num_inference_samples *
                                          num_inference_samples, 10))
        # print(samples.shape)
        results = (likelihood(samples).logits)
        # print(results.shape)
        logits_N_K_C[lower:upper].copy_(results, non_blocking=True)

        with torch.no_grad():
            candidate_batch = batchbald.get_batchbald_batch(
                logits_N_K_C.exp_(),
                acquisition_batch_size,
                num_samples,
                dtype=torch.double,
                device=device)

        targets = repeated_mnist.get_targets(
            active_learning_data.pool_dataset)
        dataset_indices = active_learning_data.get_dataset_indices(
            candidate_batch.indices)

        print("Dataset indices: ", dataset_indices)
        print("Scores: ", candidate_batch.scores)
        print("Labels: ", targets[candidate_batch.indices])

        active_learning_data.acquire(candidate_batch.indices)
        added_indices.append(dataset_indices)
        pbar.update(len(dataset_indices))

    def complete(self) -> bool:
        return False
