

from typing import Callable, List
from gpytorch.lazy.block_interleaved_lazy_tensor import BlockInterleavedLazyTensor
from gpytorch.lazy.cat_lazy_tensor import CatLazyTensor, cat
import torch
from typeguard import typechecked
from utils.typing import MultitaskMultivariateNormalType, TensorType
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from toma import toma
from tqdm import tqdm

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


def check_equal_dist(dist_1: MultitaskMultivariateNormal, dist_2: MultitaskMultivariateNormal) -> None:
    dist_1_mean = dist_1.mean
    dist_1_cov = dist_1.lazy_covariance_matrix.base_lazy_tensor.evaluate()

    dist_2_mean = dist_2.mean
    dist_2_cov = dist_2.lazy_covariance_matrix.base_lazy_tensor.evaluate()

    assert(dist_1_mean.shape == dist_2_mean.shape)
    assert(torch.allclose(dist_1_mean, dist_2_mean))

    assert(dist_1_cov.shape == dist_2_cov.shape)
    assert(torch.allclose(dist_1_cov, dist_2_cov))