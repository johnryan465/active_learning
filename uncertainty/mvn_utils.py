

from typing import Callable, List
from uncertainty.multivariate_normal import MultitaskMultivariateNormalType
from gpytorch.lazy.block_interleaved_lazy_tensor import BlockInterleavedLazyTensor
from gpytorch.lazy.cat_lazy_tensor import CatLazyTensor, cat
import torch
from typeguard import typechecked
from utils.typing import TensorType
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

            dist = MultitaskMultivariateNormalType(mean=mean, covariance_matrix=covar)
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
