from models.model import UncertainModel
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod
import torch
from batchbald.helpers import compute_entropy, compute_conditional_entropy
from batchbald.helpers import CandidateBatch


class BALD(UncertainMethod):
    def __init__(self) -> None:
        super().__init__()
        self.batch_size = 10
        self.samples = 5

    @staticmethod
    def get_bald_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None) -> CandidateBatch:
        N, K, C = log_probs_N_K_C.shape

        batch_size = min(batch_size, N)

        candidate_indices = []

        scores_N = -compute_conditional_entropy(log_probs_N_K_C)
        scores_N += compute_entropy(log_probs_N_K_C)

        candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

        return candidate_indices.tolist()

    def acquire(self, model: UncertainModel,
                dataset: ActiveLearningDataset) -> None:
        probs = model.sample(dataset.get_pool(), self.samples)
        print(probs.shape)
        indices = BALD.get_bald_batch(probs, self.batch_size)
        print(indices)
        dataset.move(indices)

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        dataset.move([i for i in range(0, 1000)])

    def complete(self) -> bool:
        return False
