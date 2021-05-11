from abc import ABC, abstractmethod
from uncertainty.current_batch import CurrentBatch
from uncertainty.multivariate_normal import MultitaskMultivariateNormalType

from batchbald_redux.batchbald import compute_conditional_entropy
from uncertainty.estimator_entropy import MVNJointEntropyEstimator, Sampling
from uncertainty.rank2 import Rank1Updates, Rank2Combine, Rank2Next
from uncertainty.mvn_utils import  chunked_distribution

from typing import Type
from typeguard import typechecked
from utils.typing import TensorType

import torch


# This class is designed to have a similar interface the the joint entropy class in batchbald_redux.
# We are wrapping the classes in estimator_entropy to wrap the complexity of the rank2 updates



class GPCEntropy(ABC):
    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Computes the entropy of this joint entropy."""
        pass

    @abstractmethod
    def add_variables(self, next: Rank2Next, selected_point: int) -> "GPCEntropy":
        """Expands the joint entropy to include more terms."""
        pass

    @abstractmethod
    def compute_batch(self, next: Rank2Combine) -> TensorType:
        """Computes the joint entropy of the added variables together with the batch (one by one)."""
        pass
    
    @staticmethod
    @typechecked
    def compute_conditional_entropy_mvn(distribution: MultitaskMultivariateNormalType, likelihood, num_samples : int) -> TensorType["N"]:
        # The distribution input is a batch of MVNS
        N = distribution.batch_shape[0]
        def func(dist: MultitaskMultivariateNormalType) -> TensorType:
            log_probs_K_n_C = (likelihood(dist.sample(sample_shape=torch.Size([num_samples]))).logits).squeeze(-2)
            log_probs_n_K_C = log_probs_K_n_C.permute(1, 0, 2)
            return compute_conditional_entropy(log_probs_N_K_C=log_probs_n_K_C)
        
        entropies_N = torch.empty(N, dtype=torch.double)
        chunked_distribution("Conditional Entropy", distribution, func, entropies_N)

        return entropies_N

class CustomEntropy(GPCEntropy):
    def __init__(self, likelihood, samples: Sampling, num_cat: int, pool_size: int, independent: MultitaskMultivariateNormalType, estimator_type: Type[MVNJointEntropyEstimator]) -> None:
        self.r2c: Rank2Combine = Rank2Combine(pool_size, independent)
        empty_batch = CurrentBatch.empty(num_cat, torch.cuda.is_available())
        self.estimator: MVNJointEntropyEstimator = estimator_type(empty_batch, likelihood, samples)

    @typechecked
    def add_variables(self, rank2: Rank2Next, selected_point: int) -> None:
        r1update = self.r2c.get_rank_1_update(selected_point)
        self.estimator.add_variable(r1update)
        self.r2c.add(rank2, selected_point)

    @typechecked
    def compute(self) -> TensorType:
        return self.estimator.compute()

    # We compute the joint entropy of the distributions
    @typechecked
    def compute_batch(self, rank2: Rank2Next) -> TensorType["N"]:
        l_rank1updates: Rank1Updates = Rank1Updates(self.r2c)
        return self.r2c.expand_to_full_pool(self.estimator.compute_batch(l_rank1updates))


