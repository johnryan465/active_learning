from abc import ABC, abstractmethod
from dataclasses import dataclass
from methods.estimator_entropy import BBReduxJointEntropyEstimator, CurrentBatch, ExactJointEntropyEstimator, MVNJointEntropyEstimator, Rank1Update, SampledJointEntropyEstimator
from methods.rank2 import Rank2Combine, Rank2Next
from methods.mvn_utils import chunked_cat_rows, chunked_distribution
from gpytorch.distributions import distribution
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from gpytorch.lazy.non_lazy_tensor import NonLazyTensor, lazify
from torch import distributions

from gpytorch.lazy import CatLazyTensor, BlockInterleavedLazyTensor, batch_repeat_lazy_tensor, cat
from torch.distributions.utils import _standard_normal
from typing import Callable, List, Type
from typeguard import typechecked
from utils.typing import MultitaskMultivariateNormalType, MultivariateNormalType, TensorType

import torch
from tqdm import tqdm
from toma import toma
import string

# This class is designed to have a similar interface the the joint entropy class in batchbald_redux.
# We are wrapping the classes in estimator_entropy to wrap the complexity of the rank2 updates



class GPCJointEntropy(ABC):
    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Computes the entropy of this joint entropy."""
        pass

    @abstractmethod
    def add_variables(self, next: Rank2Next, selected_point: int) -> "GPCJointEntropy":
        """Expands the joint entropy to include more terms."""
        pass

    @abstractmethod
    def compute_batch(self, next: Rank2Combine) -> TensorType:
        """Computes the joint entropy of the added variables together with the batch (one by one)."""
        pass


class CustomJointEntropy(GPCJointEntropy):
    def __init__(self, likelihood, samples: int, num_cat: int, pool_size: int, independent: MultitaskMultivariateNormalType, estimator_type: Type[MVNJointEntropyEstimator]) -> None:
        self.r2c: Rank2Combine = Rank2Combine(pool_size, independent)
        empty_batch = CurrentBatch.empty(num_cat)
        self.estimator: MVNJointEntropyEstimator = estimator_type(empty_batch, likelihood, samples)

    @typechecked
    def add_variables(self, rank2: Rank2Next, selected_point: int) -> None:
        compressed_idx = self.r2c.to_compressed_index(selected_point)
        r1update = self.r2c.get_rank_1_update(compressed_idx)
        self.estimator.add_variable(r1update)
        self.r2c.add(rank2, selected_point)

    @typechecked
    def compute(self) -> TensorType:
        return self.estimator.compute()

    # We compute the joint entropy of the distributions
    @typechecked
    def compute_batch(self, rank2: Rank2Next) -> TensorType["N"]:
        l_rank1updates: List[Rank1Update] = self.r2c.get_rank_1_updates()
        return self.r2c.expand_to_full_pool(self.estimator.compute_batch(l_rank1updates))

