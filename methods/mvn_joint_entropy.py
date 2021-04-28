from abc import ABC, abstractmethod
from gpytorch.distributions import distribution
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from gpytorch.lazy.non_lazy_tensor import NonLazyTensor, lazify
from torch import distributions

from gpytorch.lazy import CatLazyTensor, BlockInterleavedLazyTensor, cat
from torch._C import set_flush_denormal
from torch.distributions.utils import _standard_normal
from typing import Callable, List
from torch.tensor import Tensor
from typeguard import typechecked
from utils.typing import MultitaskMultivariateNormalType, MultivariateNormalType, TensorType
from batchbald_redux.joint_entropy import JointEntropy

import torch
from tqdm import tqdm
from toma import toma
import string




# This class wraps the rank 2 GP outputs we use for updating our joint entropy
# We are assuming that the covariance matrix is a block interleaved lazy tensor
# This enables us to perform operations in a way which give better numerical stability
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


# We need to track the rank the candiate indexes so we don't compute singular matrices
# We just want to make the ones non singular
#  | A   B |   | A   O | | I  A^{-1} B         |
#  | B^T C | = | B^T I | | 0  C - B^T A^{-1} B |
#
#  det(..)   =  det(A) det(C - B^T A^{-1} B)
# For it to be invertible we need non zero eigenvalues, the eigenvalues of (C - B^T A^{-1} B) must be non zero
# For the candidate_indexes we will replace the self covariance with I*scalar with a large positive scalar
class Rank2Combine:
    def __init__(self, pool_size: int) -> None:
        self.candidates: List[Rank2Next] = []
        self.candidate_indexes: List[int] = []
        self.pool_mask = torch.ones(pool_size, dtype=torch.bool)
        self.pool_size = pool_size

    def add(self, next_rank2: Rank2Next, index: int) -> None:
        self.candidates.append(next_rank2)
        self.candidate_indexes.append(index)
        # self.pool_mask[index] = 0

    @typechecked
    def get_mean(self) -> TensorType["N", "C"]:
        assert(len(self.candidates) > 0)
        return (self.candidates[0].get_mean())[self.pool_mask, :]

    @typechecked
    def get_self_covar(self) -> TensorType["N", "C", 1, 1]:
        assert(len(self.candidates) > 0)
        return (self.candidates[0].get_self_covar())[self.pool_mask, :]

    @typechecked
    def get_cross_mat(self) -> TensorType["N", "C", 1, "D"]:
        assert(len(self.candidates) > 0)
        return torch.cat( [nex.get_cross_mat() for nex in self.candidates], dim=3)[self.pool_mask, :]

    @typechecked
    def get_point_cross_mat(self, point: int) -> TensorType["C", 1, "D"]:
        assert(len(self.candidates) > 0)
        return torch.cat( [nex.get_cross_mat()[point] for nex in self.candidates], dim=2)

    def recreate_batch_dist(self) -> MultitaskMultivariateNormalType:
        pass

    def expand_to_full_pool(self, smaller: TensorType["M"]) -> TensorType["N"]:
        output = torch.zeros(self.pool_size)
        idx = torch.nonzero(self.pool_mask).squeeze()
        output.scatter_(0, idx, smaller)
        return output

    def to_compressed_index(self, idx: int) -> int:
        return int(torch.cumsum(self.pool_mask, dim=0)[idx].item()) - 1

    @staticmethod
    def chunked_cat_rows(A, B, C):
        N = A.shape[0]
        output = None
        chunks = []
        @toma.execute.batch(N)
        def compute(batchsize: int):
            pbar = tqdm(total=N, desc="Cat Rows", leave=False)
            start = 0
            end = 0
            chunks.clear()
            for i in range(0, N, batchsize):
                end = min(start+batchsize, N)
                A_ = A[start:end]
                B_ = B[start:end]
                C_ = C[start:end]
                
                chunk = A_.cat_rows(B_, C_)
                chunks.append(chunk)
                pbar.update(end - start)
                start = end
            pbar.close()
            
        if len(chunks) > 1:
            output = cat(chunks, dim=0)
        else:
            output = chunks[0]
        return output

# This class is designed to have a similar interface the the joint entropy class in batchbald_redux.

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

    @staticmethod
    @typechecked
    def combine_mtmvns(mvns: List[MultitaskMultivariateNormalType]) -> MultitaskMultivariateNormalType:
        if len(mvns) < 2:
            return mvns[0]

        if not all(m.event_shape == mvns[0].event_shape for m in mvns[1:]):
            raise ValueError("All MultivariateNormals must have the same event shape")

        def expand(tensor):
            if torch.is_tensor(tensor):
                return tensor.unsqueeze(0)
            else:
                return tensor._expand_batch((1,))
        # We will create a batchshape if it doesn't already exist
        if len(mvns[0].batch_shape) == 0:
            mean = cat([ expand(mvn.mean) for mvn in mvns], dim=0)
            covar_blocks_lazy = cat([expand(mvn.lazy_covariance_matrix.base_lazy_tensor) for mvn in mvns], dim=0, output_device=mean.device)
            covar_lazy = BlockInterleavedLazyTensor(covar_blocks_lazy, block_dim=-3)
        
        else:
            mean = cat([ mvn.mean for mvn in mvns], dim=0)
            covar_blocks_lazy = cat([mvn.lazy_covariance_matrix.base_lazy_tensor for mvn in mvns], dim=0, output_device=mean.device)
            covar_lazy = BlockInterleavedLazyTensor(covar_blocks_lazy, block_dim=-3)
        return MultitaskMultivariateNormal(mean=mean, covariance_matrix=covar_lazy, interleaved=True)


@typechecked
def chunked_distribution(name: str, distribution: MultitaskMultivariateNormalType, func: Callable, output: TensorType["N": ...]) -> None:
    N = output.shape[0]
    outer_batch_size = distribution.batch_shape[0]
    @toma.execute.batch(outer_batch_size)
    def compute(batchsize: int):
        pbar = tqdm(total=N, desc=name, leave=False)
        start = 0
        end = 0
        for i in range(0, outer_batch_size, batchsize):
            end = min(start+batchsize, outer_batch_size)
            mean = distribution.mean[start:end].clone()
            covar = distribution.lazy_covariance_matrix[start:end].clone()
            if torch.cuda.is_available():
                if(isinstance(mean, CatLazyTensor)):
                    mean = mean.all_to("cuda")
                else:
                    mean = mean.cuda()

                if(isinstance(covar, CatLazyTensor)):
                    covar = covar.all_to("cuda")
                else:
                    covar = covar.cuda()

            dist = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covar)
            del mean
            del covar
            g = func(dist)
            output[start:end].copy_(g, non_blocking=True)
            del dist
            del g
            pbar.update(end - start)
            start = end
        pbar.close()

# This is a class which allows to calculate the JointEntropy of GPs
# In GPytorch
class MVNJointEntropy(GPCJointEntropy):
    def __init__(self, likelihood, samples: int, num_cat: int, pool_size: int, variance_reduction: bool = False) -> None:
        self.samples: int = samples
        self.likelihood = likelihood
        self.variance_reduction = variance_reduction
        self.num_cat: int = num_cat
        # We create the batch dist as a MVN with a 0 dim size
        self.current_batch_dist: MultitaskMultivariateNormalType = MultitaskMultivariateNormal(mean=torch.zeros(0, num_cat), covariance_matrix=torch.eye(0))
        self.pool_size: int = pool_size
        self.num_points: int  = 0
        self.r2c: Rank2Combine = Rank2Combine(self.pool_size)
        self.used_points = []

    # This enables us to recompute only outputs of size 2 for our next aquisition
    @typechecked
    def join_rank_2(self, rank2: Rank2Next) -> MultitaskMultivariateNormalType[("N"),("new_batch_size","num_cat")]:
        # For each of the datapoints and the candidate batch we want to compute the low rank tensor
        # The order of the candidate datapoints must be maintained and used carefully
        # Need to wrap this in toma
        # The means will be the same for each datapoint
        if self.num_points > 0:
            item_means = self.r2c.get_mean().unsqueeze(1)
            N = item_means.shape[0]
            candidate_means = (self.current_batch_dist.mean)[None,:,:].expand(N, -1, -1)
            mean_tensor = cat([candidate_means, item_means], dim=1)

            expanded_batch_covar = self.current_batch_dist.lazy_covariance_matrix.base_lazy_tensor.expand( [N] + list(self.current_batch_dist.lazy_covariance_matrix.base_lazy_tensor.shape))
            self.self_covar_tensor = self.r2c.get_self_covar()
            cross_mat = self.r2c.get_cross_mat()
            # We need to not compute the covariance between points and themselves
            covar_tensor = Rank2Combine.chunked_cat_rows(expanded_batch_covar, cross_mat, self.self_covar_tensor)
        
        else:
            item_means = rank2.get_mean().unsqueeze(1)
            N = item_means.shape[0]
            candidate_means = (self.current_batch_dist.mean)[None,:,:].expand(N, -1, -1)
            mean_tensor = cat([candidate_means, item_means], dim=1)

            # expanded_batch_covar = self.current_batch_dist.lazy_covariance_matrix.expand( [N] + list(self.current_batch_dist.lazy_covariance_matrix.shape))
            covar_tensor = self.self_covar_tensor = rank2.get_self_covar()
        covar_tensor = BlockInterleavedLazyTensor(lazify(covar_tensor), block_dim=-3)
        return MultitaskMultivariateNormal(mean=mean_tensor, covariance_matrix=covar_tensor)

    @typechecked
    def add_variables(self, rank2: Rank2Next, selected_point: int) -> None:
        # When we are adding a new point, we update the batch
        # Get the cross correlation between the previous batches and the selected point
        if self.num_points == 0:
            _mean = rank2.get_mean()[selected_point].unsqueeze(0)
            _covar = NonLazyTensor(rank2.get_self_covar()[selected_point])
        else:
            compressed_selected_point = self.r2c.to_compressed_index(selected_point)
            cross_mat: TensorType["C", 1, "D"] = self.r2c.get_point_cross_mat(selected_point)
            self_cov: TensorType["C", 1, 1] = self.r2c.get_self_covar()[compressed_selected_point]
            new_mean: TensorType[1, "C"] = self.r2c.get_mean()[compressed_selected_point].unsqueeze(0)

            # Next we update the current distribution
            _mean = torch.cat( [self.current_batch_dist.mean, new_mean], dim=0)
            _covar = self.current_batch_dist.lazy_covariance_matrix.base_lazy_tensor.cat_rows(cross_mat, self_cov)
        _covar = BlockInterleavedLazyTensor(_covar, block_dim=-3)
        self.current_batch_dist = MultitaskMultivariateNormal(mean=_mean, covariance_matrix=_covar)
        self.r2c.add(rank2, selected_point)
        self.used_points.append(selected_point)
        self.num_points = self.num_points + 1


    @typechecked
    def compute(self) -> TensorType[1]:
        return self._compute(self.current_batch_dist, self.likelihood, self.samples, self.samples)

    @staticmethod
    def _compute(distribution: MultitaskMultivariateNormalType, likelihood, per_samples: int, total_samples: int) -> TensorType["N"]:
        # We can exactly compute a larger sized exact distribution
        # As the task batches are independent we can chunk them
        D = distribution.event_shape[0]
        C = distribution.event_shape[1]
        N = distribution.batch_shape[0]
        per_samples = per_samples
        E = total_samples # total_samples // (per_samples * D) # We could reduce the number of samples here to allow better scaling with bigger datapoints
        t = string.ascii_lowercase[:D]
        s =  ','.join(['yz' + c for c in list(t)]) + '->' + 'yz' + t
        
        @typechecked
        def exact(distribution: MultivariateNormalType) -> TensorType["N"]:
            l: TensorType["S", "N", "D", "C"] = likelihood(distribution.sample(sample_shape=torch.Size([per_samples]))).probs
            l: TensorType["N", "S", "D", "C"] = torch.transpose(l, 0, 1)
            j: List[TensorType["N", "S", "C"]] = list(torch.unbind(l, dim=-2))
            g: TensorType["N", "S", "expanded" : ...] = torch.einsum(s, *j) # We should have num_points dimensions each of size num_cat
            g: TensorType["N", "S", "E"] = torch.flatten(g, start_dim=2)
            g: TensorType["N", "E"] = torch.mean(g, 1)
            g: TensorType["N", "E"] = - g * torch.log(g)
            g: TensorType["N"] = torch.sum(g, 1)
            return g

        @typechecked
        def sampled(distribution: MultivariateNormalType) -> TensorType["N"]:
            likelihood_samples: TensorType["N", "S", "D", "C"] = torch.transpose(likelihood(distribution.sample(sample_shape=torch.Size([per_samples]))).probs, 0, 1)
            l_shape = likelihood_samples.shape
            N = l_shape[0]
            likelihood_samples: TensorType["N", "S * D", "C"] = likelihood_samples.reshape((-1, l_shape[-1]))
            # Instead of using einsum we will sample from the possible 
            # indexes, we wish to keep the same for each datapoints samples
            choices: TensorType["N", "S * D", "E"] = torch.multinomial(likelihood_samples, E, replacement=True)
            choices: TensorType["N", "S", "D", "E"] = choices.reshape( list(l_shape[:-1]) + [-1])
            likelihood_samples: TensorType["N", "S", "S", "D", "C"] = (likelihood_samples.reshape(l_shape))[:,:,None,:,:].expand(-1, -1, per_samples, -1, -1)
            choices: TensorType["N", "S", "S", "D", "E"] = choices[:,None,:,:,:].expand(-1, per_samples, -1, -1, -1)
            p: TensorType["N", "S", "S", "D", "E"] = torch.gather(likelihood_samples, 4, choices)

            p: TensorType["N", "S", "S", "D", "E"] = torch.log(p, out=p)
            p: TensorType["N", "S", "S", "E"] = torch.sum(p, dim=3)
            p: TensorType["N", "S", "S", "E"] = torch.exp(p, out=p)
            p: TensorType["N", "S", "S * E"] = p.reshape((N,per_samples,-1 ))
            p: TensorType["N", "S * E"] = torch.mean(p, 2) 
            p: TensorType["N", "S * E"] = - torch.log(p, out=p)
            p: TensorType["N"] = torch.mean(p, 1) 
            return p
        
        output = torch.zeros(N)
        if (C**D) <= E:            
            chunked_distribution("Joint Entropy", distribution, exact, output)
        else:
            chunked_distribution("Joint Entropy Sampling", distribution, sampled, output)
        return output
    # We compute the joint entropy of the distributions
    @typechecked
    def compute_batch(self, rank2: Rank2Next) -> TensorType["N"]:
        # We can exactly compute a larger sized exact distribution
        # As the task batches are independent we can chunk them
        distribution = self.join_rank_2(rank2)
        D = distribution.event_shape[0]
        N = distribution.batch_shape[0]
        C = distribution.event_shape[1]
        per_samples = self.samples
        E = self.samples # We could reduce the number of samples here to allow better scaling with bigger datapoints
        t = string.ascii_lowercase[:D]
        s =  ','.join(['yz' + c for c in list(t)]) + '->' + 'yz' + t

        if self.variance_reduction:
            # here is where we generate the samples which we will 
            shape = [per_samples] + list(distribution.base_sample_shape)
            samples = _standard_normal(torch.Size(shape), dtype=distribution.loc.dtype, device=distribution.loc.device)
        
        @typechecked
        def exact(dist: MultivariateNormalType) -> TensorType["N"]:
            N = dist.batch_shape[0]
            if self.variance_reduction:
                base_samples = samples.detach().clone()
                base_samples = base_samples[:,None,:,:]
                base_samples = base_samples.expand(-1, dist.batch_shape[0], -1, -1)
                if torch.cuda.is_available():
                    base_samples = base_samples.cuda()
                l: TensorType["S", "N", "D", "C"] = self.likelihood(dist.sample(base_samples=base_samples)).probs
            else:
                l: TensorType["S", "N", "D", "C"] = self.likelihood(dist.sample(sample_shape=torch.Size([per_samples]))).probs

            l: TensorType["N", "S", "D", "C"] = torch.transpose(l, 0, 1)
            j: List[TensorType["N", "S", "C"]] = list(torch.unbind(l, dim=-2))
            # This is where the stupid amount of memory happens
            g: TensorType["N", "S", "expanded" : ...] = torch.einsum(s, *j) # We should have num_points dimensions each of size num_cat
            g: TensorType["N", "S", "E"] = torch.flatten(g, start_dim=2)
            g: TensorType["N", "E"] = torch.mean(g, 1)
            g: TensorType["N", "E"] = - g * torch.log(g)
            g: TensorType["N"] = torch.sum(g, 1)
            return g

        @typechecked
        def sampled(distribution: MultivariateNormalType) -> TensorType["N"]:
            N = distribution.batch_shape[0]
            if self.variance_reduction:
                base_samples = samples.detach().clone()
                base_samples = base_samples[:,None,:,:]
                base_samples = base_samples.expand(-1, distribution.batch_shape[0], -1, -1)
                likelihood_samples: TensorType["S", "N", "D", "C"] = self.likelihood(distribution.sample(base_samples=base_samples)).probs
                del base_samples
            else:
                likelihood_samples: TensorType["S", "N", "D", "C"] = self.likelihood(distribution.sample(sample_shape=torch.Size([per_samples]))).probs

            likelihood_samples: TensorType["N", "S", "D", "C"] = torch.transpose(likelihood_samples, 0, 1)
            l_shape = likelihood_samples.shape
            likelihood_samples: TensorType["N * S * D", "C"] = likelihood_samples.reshape((-1, l_shape[-1]))
            # Instead of using einsum we will sample from the possible 
            # indexes, we wish to keep the same for each datapoints samples
            choices: TensorType["N * S * D", "E"] = torch.multinomial(likelihood_samples, E, replacement=True)
            choices: TensorType["N", "S", "D", "E"] = choices.reshape( list(l_shape[:-1]) + [-1])
            likelihood_samples: TensorType["N", "S", "S", "D", "C"] = (likelihood_samples.reshape(l_shape))[:,:,None,:,:].expand(-1, -1, per_samples, -1, -1)
            choices: TensorType["N", "S", "S", "D", "E"] = choices[:,None,:,:,:].expand(-1, per_samples, -1, -1, -1)
            p: TensorType["N", "S", "S", "D", "E"] = torch.gather(likelihood_samples, 4, choices)
            del choices
            del likelihood_samples

            p: TensorType["N", "S", "S", "D", "E"] = torch.log(p, out=p)
            p: TensorType["N", "S", "S", "E"] = torch.sum(p, dim=3)
            p: TensorType["N", "S", "S", "E"] = torch.exp(p, out=p)
            # The mean needs to be rescaled
            p: TensorType["N", "S", "S * E"] = p.reshape((N,per_samples,-1 ))
            p: TensorType["N", "S * E"] = torch.mean(p, 1) 
            p: TensorType["N", "S * E"] = - torch.log(p, out=p)
            p: TensorType["N"] = torch.mean(p, 1) 
            return p
        
        output = torch.zeros(N)
        if (C**D) <= E:            
            chunked_distribution("Joint Entropy", distribution, exact, output)
        else:
            chunked_distribution("Joint Entropy Sampling", distribution, sampled, output)
        return self.r2c.expand_to_full_pool(output)


class LowMemMVNJointEntropy(GPCJointEntropy):
    # The difference between this class and the one above is that we are first sampling from current
    # batch, and using this to create conditional distirubitons for each of our new datapoints.
    # This leads to a reduction in memory usage as we are sampling from a distirbution of size D * C,
    # Creating a distribution of size 1 * C, sampling from it and combining both to get our results.
    # This is in comparison to sampling from a distribtuion of size (D+1) * C for each datapoint.

    #  |x|     | |mu_X|   | sigma_XX sigma_XY | |
    #  |Y| = N | |mu_Y| , | sigma_YX sigma_YY | |
    #
    #
    # Y|X = N( mu_Y + sigma_YX sigma_XX^{-1} (X - mu_x), sigma_YY - sigma_{YX} sigma_{XX}^{-1} sigma_{XY} )
    # 
    # 1) The conditional distributions covariance is independent of the particular batch sample
    # 2) The means are not
    # ["N", "L", "C", "B"]
    # Given a function sample and a conditional distribution we wish to compute the entropy
    # The function sample is a (D * C) tensor
    # The distribution is a 1*C normal distribution
    # We sample the current batch distribution w times. we can use the same gather mat
    def __init__(self, likelihood, sum_samples: int, batch_samples: int, per_samples: int, num_cat: int, pool_size: int, variance_reduction: bool = False) -> None:
        self.likelihood = likelihood
        self.variance_reduction = variance_reduction
        self.num_cat: int = num_cat
        # We create the batch dist as a MVN with a 0 dim size
        self.current_batch_dist: MultitaskMultivariateNormalType = MultitaskMultivariateNormal(mean=torch.zeros(0, num_cat), covariance_matrix=torch.eye(0))
        self.pool_size: int = pool_size
        self.num_points: int  = 0
        self.r2c: Rank2Combine = Rank2Combine(self.pool_size)
        self.used_points = []
        self.per_sample: int = per_samples
        self.batch_samples: int = batch_samples
        self.sum_samples: int = sum_samples

    # This enables us to recompute only outputs of size 2 for our next aquisition
    @typechecked
    def join_rank_2(self, rank2: Rank2Next) -> MultitaskMultivariateNormalType[("N"),("new_batch_size","num_cat")]:
        # For each of the datapoints and the candidate batch we want to compute the low rank tensor
        # The order of the candidate datapoints must be maintained and used carefully
        # Need to wrap this in toma
        # The means will be the same for each datapoint
        if self.num_points > 0:
            item_means = self.r2c.get_mean().unsqueeze(1)
            N = item_means.shape[0]
            candidate_means = (self.current_batch_dist.mean)[None,:,:].expand(N, -1, -1)
            mean_tensor = cat([candidate_means, item_means], dim=1)

            expanded_batch_covar = self.current_batch_dist.lazy_covariance_matrix.base_lazy_tensor.expand( [N] + list(self.current_batch_dist.lazy_covariance_matrix.base_lazy_tensor.shape))
            self.self_covar_tensor = self.r2c.get_self_covar()
            cross_mat = self.r2c.get_cross_mat()
            # We need to not compute the covariance between points and themselves
            covar_tensor = Rank2Combine.chunked_cat_rows(expanded_batch_covar, cross_mat, self.self_covar_tensor)
        
        else:
            item_means = rank2.get_mean().unsqueeze(1)
            N = item_means.shape[0]
            candidate_means = (self.current_batch_dist.mean)[None,:,:].expand(N, -1, -1)
            mean_tensor = cat([candidate_means, item_means], dim=1)

            expanded_batch_covar = self.current_batch_dist.lazy_covariance_matrix.expand( [N] + list(self.current_batch_dist.lazy_covariance_matrix.shape))
            covar_tensor = self.self_covar_tensor = rank2.get_self_covar()
        covar_tensor = BlockInterleavedLazyTensor(lazify(covar_tensor), block_dim=-3)
        return MultitaskMultivariateNormal(mean=mean_tensor, covariance_matrix=covar_tensor)

    @typechecked
    def add_variables(self, rank2: Rank2Next, selected_point: int) -> None:
        # When we are adding a new point, we update the batch
        # Get the cross correlation between the previous batches and the selected point
        if self.num_points == 0:
            _mean = rank2.get_mean()[selected_point].unsqueeze(0)
            _covar = lazify(rank2.get_self_covar()[selected_point])
        else:
            compressed_selected_point = self.r2c.to_compressed_index(selected_point)
            cross_mat: TensorType["C", 1, "D"] = self.r2c.get_point_cross_mat(selected_point)
            self_cov: TensorType["C", 1, 1] = self.r2c.get_self_covar()[compressed_selected_point]
            new_mean: TensorType[1, "C"] = self.r2c.get_mean()[compressed_selected_point].unsqueeze(0)

            # Next we update the current distribution
            _mean = torch.cat( [self.current_batch_dist.mean, new_mean], dim=0)
            _covar = self.current_batch_dist.lazy_covariance_matrix.base_lazy_tensor.cat_rows(cross_mat, self_cov)
        _covar = BlockInterleavedLazyTensor(lazify(_covar), block_dim=-3)
        if torch.cuda.is_available():
            _mean = _mean.cuda()
            _covar = _covar.cuda()
        
        self.current_batch_dist = MultitaskMultivariateNormal(mean=_mean, covariance_matrix=_covar)
        self.r2c.add(rank2, selected_point)
        self.used_points.append(selected_point)
        self.num_points = self.num_points + 1
        self.create_samples(self.sum_samples, self.batch_samples)


    # We call this function to create the samples for the batch distribution
    # We use r2c and the current batch dist to create the samples and conditional distributions
    def create_samples(self, sum_samples: int, batch_samples: int) -> None:
        distribution = self.current_batch_dist
        likelihood = self.likelihood

        likelihood_samples: TensorType["S", "D", "C"] = likelihood(distribution.sample(sample_shape=torch.Size([batch_samples]))).probs


        # The covariance is independent of the function valuees
        # covar = sigma_YY - sigma_{YX} sigma_{XX}^{-1} sigma_{XY}
        sigma_YY: TensorType["N", "C", 1, 1] = self.r2c.get_self_covar()
        sigma_YX: TensorType["N", "C", 1, "D"] = self.r2c.get_cross_mat()
        sigma_XX: TensorType["C", "D", "D"] = self.current_batch_dist.lazy_covariance_matrix.base_lazy_tensor.evaluate()
        sigma_XX_inv: TensorType["C", "D", "D"] = torch.inverse(sigma_XX)

        N = sigma_YY.shape[0]

        conditional_cov: TensorType["N", "C", 1, 1] = sigma_YY - (sigma_YX @ sigma_XX_inv @ torch.transpose( sigma_YX, -1, -2))

        #  mean = mu_Y + sigma_YX sigma_XX^{-1} (X - mu_x)
        # We want to expand the batch dist mean and covariance so we can broadcast in number of samples
        mu_Y: TensorType["N", "S", "C", 1] = (self.r2c.get_mean()[:,None,:,None].expand(-1, batch_samples, -1, -1))
        mu_X: TensorType["N", "S", "C", "D"] = torch.transpose(self.current_batch_dist.mean[None, None, :, :].expand(N, batch_samples, -1, -1), -1, -2)
        X: TensorType["N", "S", "C", "D"] = torch.transpose(likelihood_samples[None,:,:,:].expand(N ,-1, -1, -1), -1, -2)

        if torch.cuda.is_available():
            X = X.cuda()

        tmp_vector: TensorType["N", "S", "C", "D", 1] = (X - mu_X).unsqueeze(-1)

        # We can let pytorch auto broadcast the matrix multiplication
        tmp_matrix: TensorType["N", "S", "C", "D" ,"D"] = (sigma_YX @ sigma_XX_inv).unsqueeze(1).expand(-1, batch_samples, -1, -1, -1)
        conditional_mean: TensorType["N", "S", "C", 1, 1] = (tmp_matrix @ tmp_vector)
        conditional_mean =   mu_Y.unsqueeze(-1) + conditional_mean

        l_shape = likelihood_samples.shape
        likelihood_samples: TensorType["S * D", "C"] = likelihood_samples.reshape((-1, l_shape[-1]))
        # Instead of using einsum we will sample from the possible 
        # indexes, we wish to keep the same for each datapoints samples

        
        choices: TensorType["S * D", "E"] = torch.multinomial(likelihood_samples, sum_samples, replacement=True)
        choices: TensorType["S", "D", "E"] = choices.reshape( list(l_shape[:-1]) + [-1])
        likelihood_samples: TensorType["S", "D", "C"] = likelihood_samples.reshape(l_shape)
        
        if True:
            choices: TensorType["S", "D", "E"] = choices.reshape( list(l_shape[:-1]) + [-1])
            likelihood_samples: TensorType["S", "D", "C"] = likelihood_samples.reshape(l_shape)


            likelihood_samples: TensorType["S", "S", "D", "C"] = likelihood_samples[:,None,:,:].expand(-1, batch_samples, -1, -1)
            choices: TensorType["S", "S", "D", "E"] = choices[None,:,:,:].expand(batch_samples, -1, -1, -1)
            p: TensorType["S", "S", "D", "E"] = torch.gather(likelihood_samples, 3, choices)


            p: TensorType["S", "S", "D", "E"] = torch.log(p, out=p)
            p: TensorType["S", "S", "E"] = torch.sum(p, dim=2) # For each of the samples we have a random sample of log probs
            p: TensorType["S", "S*E"] = torch.flatten(p, start_dim=1)
        else:
            p: TensorType["S", "D", "E"] = torch.gather(likelihood_samples, 2, choices)


            p: TensorType["S", "D", "E"] = torch.log(p, out=p)
            p: TensorType["S", "E"] = torch.sum(p, dim=1) # For each of the samples we have a random sample of log probs
            # p: TensorType["S", "S*E"] = torch.flatten(p, start_dim=1)

        conditional_cov: TensorType["N", "S", "C", 1, 1] = conditional_cov[:,None,:,:,:].expand(-1, batch_samples, -1, -1 , -1)
        conditional_mean: TensorType["N", "S", "C", 1] = conditional_mean.squeeze(-1)
        self.conditional_mean = conditional_mean
        self.conditional_cov = BlockInterleavedLazyTensor(NonLazyTensor(conditional_cov), block_dim=-3)
        self.conditional_dist = MultitaskMultivariateNormal(mean=self.conditional_mean, covariance_matrix=self.conditional_cov, interleaved=True)
        self.log_probs = p

    @typechecked
    def compute(self) -> TensorType[1]:
        return MVNJointEntropy._compute(self.current_batch_dist, self.likelihood, self.per_sample, self.per_sample)


    @typechecked
    def compute_batch(self, rank2: Rank2Next) -> TensorType["N"]:
        # We can exactly compute a larger sized exact distribution
        # As the task batches are independent we can chunk them
        # If we haven't added any variables yet, the coditional doesn't exist yet, but we have
        # nothing to conditon on. We have only 1 "sample", which has a probability of 1
        if self.num_points == 0:
            means: TensorType["N", 1, "C"] = rank2.get_mean()[:, None,:, None]
            covar: TensorType["N", 1, "C", 1, 1] = rank2.get_self_covar().unsqueeze(1)
            covar = BlockInterleavedLazyTensor(lazify(covar), block_dim=-3)
            distribution = MultitaskMultivariateNormal(mean=means, covariance_matrix=covar)
            log_p = torch.tensor([[0]]) # log(1) = 0
        else:
            distribution = self.conditional_dist
            log_p = self.log_probs

        if torch.cuda.is_available():
            log_p = log_p.cuda()
        D = distribution.event_shape[0]
        N = distribution.batch_shape[0]
        C = distribution.event_shape[1]
        per_samples = self.per_sample

        # We are sampling more over the batch than than the candiadate points
        @typechecked
        def sampled(distribution: MultivariateNormalType) -> TensorType["N"]:
            # We have L samples from the batch distribution
            # We have B samples from the sum of the batch distribution
            # We have P samples from the conditional distribution
            P = per_samples
            # We have N points on the distribution
            # We have E samples from the sum of the conditional
            N = distribution.batch_shape[0]
            likelihood_samples: TensorType["P", "N", "L", "C"] = (self.likelihood(distribution.sample(sample_shape=torch.Size([P]))).probs).squeeze(-2)

            likelihood_samples: TensorType["N", "P", "L", "C"] = torch.transpose(likelihood_samples, 0, 1)
            p_y: TensorType["N", "L", "C"] = torch.mean(likelihood_samples, dim=1)
            batch_p: TensorType["L", "B"] = log_p.exp()
            p_expanded: TensorType["N", "L", "C", 1] = p_y.unsqueeze(-1)
            batch_p_expanded: TensorType["L", 1, "B"] = batch_p.unsqueeze(-2)
            # As we are not sampling from the indexes of the conditional, we need to rescale
            # to make it an accurate estimator
            # The individual probabilites are accurate
            # Without reweighting we would get H(X,Y) = - \sum_X \sum_Y p(x) log p(x,y) 
            # p(x)p(y|x) = p(x,y)
            p: TensorType["N", "L", "C", "B"] = p_expanded * batch_p_expanded
            p: TensorType["N", "C", "B"] = torch.mean(p, dim=1) # p(x,y)
            batch_p: TensorType["B"] = torch.mean(batch_p, dim = 0)# p(x)

            p: TensorType["N", "B"] = torch.sum(- (p / batch_p) * torch.log(p), dim = 1)
            # p: TensorType["N", "B"] = p / batch_p
            p: TensorType["N"] = torch.mean(p, 1) 
            return p
        
        output = torch.zeros(N)
        chunked_distribution("Joint Entropy Sampling", distribution, sampled, output)
        # print("output")
        # print(output)
        o = self.r2c.expand_to_full_pool(output)
        # print(o)
        return o
