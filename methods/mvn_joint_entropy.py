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
import gc

def debug_gpu():
    # Debug out of memory bugs.
    # tensor_list = []
    tensor_count = 0
    for obj in gc.get_objects():
        try:
            # print(obj.__name__)
            if torch.is_tensor(obj):# or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor_count = tensor_count + 1
                print(type(obj), obj.size())
        except:
            pass
    print(f'Count of tensors = {tensor_count}.')

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
class MVNJointEntropy:
    def __init__(self, current_batch_dist: MultitaskMultivariateNormalType, rank_2_distributions: MultitaskMultivariateNormalType, samples: int) -> None:
        cov = rank_2_distributions.lazy_covariance_matrix.evaluate()
        self.num_cats = rank_2_distributions.event_shape[1]
        self.current_batch_dist = current_batch_dist # We replace this on adding variables
        self.samples = samples
        self.rank_2_distributions_mean = rank_2_distributions.mean[:,0,1,:].unsqueeze(1)  # This wont change
        self.self_covar_tensor = cov[:, 0 , self.num_cats:, self.num_cats:]               # This wont change
        self.cross_mat = cat( torch.unbind(cov[:, :, :self.num_cats, self.num_cats:], dim=1), dim=-1) # We will concat
        self.num_datapoints = rank_2_distributions.batch_shape[0]

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

    # This enables us to recompute only outputs of size 2 for our next aquisition
    @typechecked
    def join_rank_2(self) -> MultitaskMultivariateNormalType[("N"),("new_batch_size","num_cat")]:
        # For each of the datapoints and the candidate batch we want to compute the low rank tensor
        # The order of the candidate datapoints must be maintained and used carefully
        # Need to wrap this in toma
        # The means will be the same for each datapoint
        item_means = self.rank_2_distributions_mean
        candidate_means = (self.current_batch_dist.mean).expand(self.num_datapoints, -1, -1)
        mean_tensor = cat([candidate_means, item_means], dim=1)

        expanded_batch_covar = self.current_batch_dist.lazy_covariance_matrix.expand( [self.num_datapoints] + list(self.current_batch_dist.lazy_covariance_matrix.shape[1:]))
        self_covar_tensor = self.self_covar_tensor
        cross_mat = self.cross_mat
        # debug_gpu()
        covar_tensor = MVNJointEntropy.chunked_cat_rows(expanded_batch_covar, cross_mat, self_covar_tensor).cpu()

        return MultitaskMultivariateNormal(mean=mean_tensor, covariance_matrix=covar_tensor)

    @staticmethod
    @typechecked
    def combine_mtmvns(mvns) -> MultitaskMultivariateNormalType:
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
            covar_blocks_lazy = cat([expand(mvn.lazy_covariance_matrix) for mvn in mvns], dim=0, output_device=mean.device)
            covar_lazy = BlockDiagLazyTensor(covar_blocks_lazy, block_dim=0)
        
        else:
            mean = cat([ mvn.mean for mvn in mvns], dim=0)
            covar_blocks_lazy = cat([mvn.lazy_covariance_matrix for mvn in mvns], dim=0, output_device=mean.device)
            covar_lazy = BlockDiagLazyTensor(covar_blocks_lazy, block_dim=0)
        return MultitaskMultivariateNormal(mean=mean, covariance_matrix=covar_blocks_lazy, interleaved=True)

    @typechecked
    def add_new(self, new_batch: MultitaskMultivariateNormalType, rank_2_combinations: MultitaskMultivariateNormalType) -> None:
        # When we are adding a new point, we update the batch
        # And we updated the cross_mat
        cov = rank_2_combinations.lazy_covariance_matrix.evaluate()
        self.current_batch_dist = new_batch
        self.cross_mat = cat( [self.cross_mat, cov[:, 0, :self.num_cats, self.num_cats:]], dim=-1) 

    # We compute the joint entropy of the distributions
    @staticmethod
    @typechecked
    def compute(distribution: MultivariateNormalType, likelihood, per_samples: int, total_samples: int, output: TensorType["N"], variance_reduction: bool = False) -> None:
        # We can exactly compute a larger sized exact distribution
        # As the task batches are independent we can chunk them
        D = distribution.event_shape[0]
        N = distribution.batch_shape[0]
        C = distribution.event_shape[1]
        per_samples = per_samples // D
        E = total_samples // (per_samples * D) # We could reduce the number of samples here to allow better scaling with bigger datapoints
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
                if torch.cuda.is_available():
                    base_samples = base_samples.cuda()
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
                del base_samples
            else:
                likelihood_samples: TensorType["S", "N", "D", "C"] = likelihood(distribution.sample(sample_shape=torch.Size([per_samples]))).probs

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
            
        if (C**D) >= E:            
            chunked_distribution("Joint Entropy", distribution, exact, output)
        else:
            chunked_distribution("Joint Entropy Sampling", distribution, sampled, output)
