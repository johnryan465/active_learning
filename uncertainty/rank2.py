
# This class wraps the rank 2 GP outputs we use for updating our joint entropy
# We are assuming that the covariance matrix is a block interleaved lazy tensor
# This enables us to perform operations in a way which give better numerical stability
from uncertainty.estimator_entropy import Rank1Update
from typing import List
import torch
from utils.typing import MultitaskMultivariateNormalType, TensorType
from typeguard import typechecked
from toma import toma
from tqdm import tqdm
from gpytorch.lazy import cat


# Here we encapsilate the logic for allowing us to minimise the amount of 
# wasted computation at each iteration.

class Rank2Next:
    # We interprete the first item as the candidate item
    def __init__(self, rank_2_mvn: MultitaskMultivariateNormalType[("N",1),(2, "C")]) -> None:
        self.cov = rank_2_mvn.lazy_covariance_matrix.base_lazy_tensor.evaluate().squeeze(1)
        self.mean = (rank_2_mvn.mean.squeeze(1))[:, 1, :]

    @typechecked
    def get_mean(self) -> TensorType["N", "C"]: # mu_p
        return self.mean.clone()

    @typechecked
    def get_self_covar(self) -> TensorType["N", "C", 1, 1]: # k(p,p)
        return self.cov[:, : , 1:, 1:].clone()

    @typechecked
    def get_candidate_covar(self) -> TensorType["C", 1]: # k(c,c)
        return self.cov[0, :, :1, :1].clone()
    
    @typechecked
    def get_cross_mat(self) -> TensorType["N", "C", 1, 1]:
        return self.cov[:, :, :1, 1:].clone()


class Rank2Combine:
    def __init__(self, pool_size: int, independent: MultitaskMultivariateNormalType) -> None:
        self.candidates: List[Rank2Next] = []
        self.candidate_indexes: List[int] = []
        self.pool_mask = torch.ones(pool_size, dtype=torch.bool)
        self.pool_size = pool_size
        self.mean = independent.mean.squeeze(1)
        self.self_covar = independent.lazy_covariance_matrix.base_lazy_tensor.evaluate()

    def add(self, next_rank2: Rank2Next, index: int) -> None:
        self.candidates.append(next_rank2)
        self.candidate_indexes.append(index)
        self.pool_mask[index] = 0

    @typechecked
    def get_mean(self) -> TensorType["N", "C"]:
        return (self.mean)[self.pool_mask, :]

    @typechecked
    def get_self_covar(self) -> TensorType["N", "C", 1, 1]:
        return (self.self_covar)[self.pool_mask, :]

    @typechecked
    def get_cross_mat(self) -> TensorType["N", "C", 1, "D"]:
        if len(self.candidates) > 0:
            return torch.cat( [nex.get_cross_mat() for nex in self.candidates], dim=3)[self.pool_mask, :]
        else:
            N = self.mean.shape[0]
            C = self.mean.shape[1]
            return torch.zeros(N,C,1,0)

    @typechecked
    def get_point_cross_mat(self, point: int) -> TensorType["C", 1, "D"]:
        assert(len(self.candidates) > 0)
        return torch.cat( [nex.get_cross_mat()[point] for nex in self.candidates], dim=2)

    def expand_to_full_pool(self, smaller: TensorType["M"]) -> TensorType["N"]:
        output = torch.zeros(self.pool_size)
        idx = torch.nonzero(self.pool_mask).squeeze()
        output.scatter_(0, idx, smaller)
        return output

    def to_compressed_index(self, idx: int) -> int:
        return int(torch.cumsum(self.pool_mask, dim=0)[idx].item()) - 1

    def get_rank_1_update(self, idx: int) -> Rank1Update:
        mean = self.get_mean()[idx].unsqueeze(0)
        self_cov = self.get_self_covar()[idx]
        cross_mat = self.get_cross_mat()[idx]
        return Rank1Update(mean, self_cov, cross_mat)

    def get_rank_1_updates(self) -> List[Rank1Update]:
        updates = []
        for idx in range(0, self.pool_size - len(self.candidate_indexes)):
            updates.append(self.get_rank_1_update(idx))
        return updates