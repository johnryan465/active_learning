
# This class wraps the rank 2 GP outputs we use for updating our joint entropy
# We are assuming that the covariance matrix is a block interleaved lazy tensor
# This enables us to perform operations in a way which give better numerical stability
from dataclasses import dataclass
from typing import Iterator, List
import torch
from utils.typing import MultitaskMultivariateNormalType, TensorType
from typeguard import typechecked
from toma import toma
from tqdm import tqdm
from gpytorch.lazy import cat



@dataclass
class Rank1Update:
    mean: TensorType
    covariance: TensorType
    cross_covariance: TensorType

# Rank1Updates = Iterator[Rank1Update]


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

class Rank1Updates:
    def __init__(self, r2c: Rank2Combine = None, already_computed: List[Rank1Update] = None):
        if already_computed is not None:
            self.max = len(already_computed)
            self.values = already_computed
            self.already_computed = True
        else:
            self.max = r2c.pool_size - len(r2c.candidate_indexes)
            self.r2c = r2c
            self.already_computed = False
        self.idx = 0
        self.size = self.max

    def __iter__(self):
        return self
    
    def __len__(self):
        return self.size

    def reset(self):
        self.idx = 0

    def __next__(self) -> Rank1Update:
        if self.idx < self.max:
            if self.already_computed:
                update = self.values[self.idx]
            else:
                update = self.r2c.get_rank_1_update(self.idx)
            self.idx += 1
            return update
        else:
            raise StopIteration
