from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from torch import distributions

from gpytorch.lazy import CatLazyTensor, BlockDiagLazyTensor, cat
from torch.distributions.utils import _standard_normal
from typing import Callable, List
from typeguard import typechecked
from utils.typing import MultitaskMultivariateNormalType, MultivariateNormalType, TensorType

import torch
from tqdm import tqdm
from toma import toma
import string


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
            mean = distribution.mean[start:end]
            covar = distribution.lazy_covariance_matrix[start:end]
            dist = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covar)
            g = func(dist)
            output[start:end].copy_(g, non_blocking=True)
            del g
            pbar.update(end - start)
            start = end
            pbar.close()

# This is a class which allows to calculate the JointEntropy of GPs
# In GPytorch
class MVNJointEntropy:
    def __init__(self, current_batch_dist: MultitaskMultivariateNormalType, samples: int) -> None:
        self.current_batch_dist = current_batch_dist
        self.samples = samples
        self.rank_2_distributions = []

    # This enables us to recompute only outputs of size 2 for our next aquisition
    @typechecked
    def join_rank_2(self, rank_2_dists: MultitaskMultivariateNormalType[("datapoints","batch_size"), (2, "num_cat")]) -> MultitaskMultivariateNormalType[("N"),("new_batch_size","num_cat")]:
        # For each of the datapoints and the candidate batch we want to compute the low rank tensor
        # The order of the candidate datapoints must be maintained and used carefully
        # Need to wrap this in toma
        batch_size = self.current_batch_dist.event_shape[0]
        num_cats = self.current_batch_dist.event_shape[1]
        num_datapoints = rank_2_dists.batch_shape[0]
        cov = rank_2_dists.covariance_matrix

        item_means = rank_2_dists.mean[:,1,1,:].unsqueeze(1)
        candidate_means = self.current_batch_dist.mean[None,:,:].expand(num_datapoints, -1, -1)
        mean_tensor = torch.cat([candidate_means, item_means], dim=1)

        expanded_batch_covar = self.current_batch_dist.lazy_covariance_matrix.expand(num_datapoints, batch_size*num_cats, batch_size*num_cats)
        self_covar_tensor = cov[:, 0, num_cats:, num_cats:]
        cross_mat = torch.cat(torch.unbind(cov[:,:, :num_cats, num_cats:], dim=1), dim=-1)
        covar_tensor = expanded_batch_covar.cat_rows(cross_mat, self_covar_tensor)

        return MultitaskMultivariateNormal(mean=mean_tensor, covariance_matrix=covar_tensor)

    @staticmethod
    @typechecked
    def combine_mtmvns(mvns) -> MultitaskMultivariateNormalType:
        if len(mvns) < 2:
            return mvns[0]

        if not all(m.event_shape == mvns[0].event_shape for m in mvns[1:]):
            raise ValueError("All MultivariateNormals must have the same event shape")
        mean = cat([mvn.mean for mvn in mvns], dim=0)

        covar_blocks_lazy = CatLazyTensor(
            *[mvn.lazy_covariance_matrix for mvn in mvns], dim=0, output_device=mean.device
        )
        covar_lazy = BlockDiagLazyTensor(covar_blocks_lazy, block_dim=0)
        return MultitaskMultivariateNormal(mean=mean, covariance_matrix=covar_blocks_lazy, interleaved=False)

    @typechecked
    def add_new(self, new_batch: MultitaskMultivariateNormalType, rank_2_combinations: MultitaskMultivariateNormalType) -> None:
        self.current_batch_dist = new_batch
        self.rank_2_distributions.append(rank_2_combinations)

    # We compute the joint entropy of the distributions
    @typechecked
    def compute(self, distribution: MultivariateNormalType, likelihood, S: int, output: TensorType["N"], variance_reduction: bool = False) -> None:
        # We can exactly compute a larger sized exact distribution
        # As the task batches are independent we can chunk them
        D = distribution.event_shape[0]
        N = distribution.batch_shape[0]
        C = distribution.event_shape[1]
        S = 100
        E = 10000 // S
        per_samples = S
        t = string.ascii_lowercase[:D]
        s =  ','.join(['yz' + c for c in list(t)]) + '->' + 'yz' + t

        if variance_reduction:
            # here is where we generate the samples which we will 
            shape = [per_samples] + list(distribution.base_sample_shape)
            samples = _standard_normal(torch.Size(shape), dtype=distribution.loc.dtype, device=distribution.loc.device)
        
        @typechecked
        def exact(distribution: MultivariateNormalType) -> TensorType["N"]:
            N = distribution.batch_shape[0]
            if variance_reduction:
                base_samples = samples.detach().clone()
                base_samples = base_samples[:,None,:,:]
                base_samples = base_samples.expand(-1, distribution.batch_shape[0], -1, -1)
                l: TensorType["S", "N", "D", "C"] = likelihood(distribution.sample(base_samples=base_samples)).probs
            else:
                l: TensorType["S", "N", "D", "C"] = likelihood(distribution.sample(sample_shape=torch.Size([per_samples]))).probs

            l: TensorType["N", "S", "D", "C"] = torch.transpose(l, 0, 1)
            j: List[TensorType["N", "S", "C"]] = list(torch.unbind(l, dim=-2))
            # This is where the stupid amount of memory happens
            g: TensorType["N", "S", "expanded" : ...] = torch.einsum(s, *j) # We should have num_points dimensions each of size num_cat
            g: TensorType["N", "S", "E"] = torch.flatten(g, start_dim=2)
            g: TensorType["N", "E"] = torch.mean(g, 1)
            #
            g: TensorType["N", "E"] = - g * torch.log(g)
            g: TensorType["N"] = torch.sum(g, 1)
            return g

        @typechecked
        def sampled(distribution: MultivariateNormalType) -> TensorType["N"]:
            N = distribution.batch_shape[0]
            if variance_reduction:
                base_samples = samples.detach().clone()
                base_samples = base_samples[:,None,:,:]
                base_samples = base_samples.expand(-1, distribution.batch_shape[0], -1, -1)
                likelihood_samples: TensorType["S", "N", "D", "C"] = likelihood(distribution.sample(base_samples=base_samples)).probs
            else:
                likelihood_samples: TensorType["S", "N", "D", "C"] = likelihood(distribution.sample(sample_shape=torch.Size([per_samples]))).probs

            likelihood_samples: TensorType["N", "S", "D", "C"] = torch.transpose(likelihood_samples, 0, 1)
            likelihood_expanded: TensorType["N", "S", "S", "D", "C"] = likelihood_samples[:,:,None,:,:].expand(-1, -1, per_samples, -1, -1)
            l_shape = likelihood_samples.shape
            # Instead of using einsum we will sample from the possible 
            # indexes, we wish to keep the same for each datapoints samples
            y: TensorType["N * S * D", "C"] = likelihood_samples.reshape((-1, l_shape[-1]))
            choices: TensorType["N * S * D", "E"] = torch.multinomial(y, E, replacement=True)
            choices: TensorType["N", "S", "D", "E"] = choices.reshape( list(l_shape[:-1]) + [-1])
            choices_expanded: TensorType["N", "S", "S", "D", "E"] = choices[:,None,:,:,:].expand(-1, per_samples, -1, -1, -1)

            w: TensorType["N", "S", "S", "D", "E"] = torch.gather(likelihood_expanded, 4, choices_expanded)
            # g: TensorType["N", "S", "E"] = torch.prod(w, 2)
            p: TensorType["N", "S", "S", "E"] = torch.exp(torch.sum(torch.log(w), dim=3))

            # The mean needs to be rescaled
            p: TensorType["N", "S", "S * E"] = p.reshape((N,S,-1 ))
            p: TensorType["N", "S * E"] = torch.mean(p, 1) 
            p: TensorType["N", "S * E"] = - torch.log(p)
            p: TensorType["N"] = torch.mean(p, 1) 
            return p
            
        if S * (C**D)  <= 10000:            
            chunked_distribution("Joint Entropy", distribution, exact, output)
        else:
            chunked_distribution("Joint Entropy Sampling", distribution, sampled, output)
