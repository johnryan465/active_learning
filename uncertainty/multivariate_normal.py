from typing import Annotated, List
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.lazy.block_interleaved_lazy_tensor import BlockInterleavedLazyTensor
from gpytorch.lazy.cat_lazy_tensor import cat
from gpytorch.lazy.non_lazy_tensor import lazify
import torch
from utils.typing import TensorType
from torch.tensor import Tensor


class _MultitaskMultivariateNormalType(MultitaskMultivariateNormal):
    def __init__(self, mean, covariance_matrix, validate_args=False, interleaved=True):
        super().__init__(mean, covariance_matrix, validate_args=validate_args, interleaved=interleaved)

    def __eq__(self, other):
        if isinstance(other, _MultitaskMultivariateNormalType):
            mean_1 = self.mean
            mean_2 = other.mean
            covar_1 = self.lazy_covariance_matrix.evaluate()
            covar_2 = other.lazy_covariance_matrix.evaluate()
            return (mean_1.shape == mean_2.shape) and (covar_1.shape == covar_2.shape) and torch.allclose(mean_1, mean_2) and torch.allclose(covar_1, covar_2)
        return False

    @staticmethod
    def combine_mtmvns(mvns: List) -> "_MultitaskMultivariateNormalType":
        if len(mvns) < 2:
            mean = mvns[0].mean
            interleaved = mvns[0]._interleaved
            covar = mvns[0].lazy_covariance_matrix
            return _MultitaskMultivariateNormalType(mean=mean, covariance_matrix=covar , interleaved=interleaved)

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
        return _MultitaskMultivariateNormalType(mean=mean, covariance_matrix=covar_lazy, interleaved=True)

    @staticmethod
    def create(mean: TensorType, covar: TensorType, cuda: bool) -> "_MultitaskMultivariateNormalType":
        if cuda:
            mean = mean.cuda()
            covar = lazify(covar).cuda()
        else:
            covar = lazify(covar)
        lazy_covar = BlockInterleavedLazyTensor(covar, block_dim=-3)
        return _MultitaskMultivariateNormalType(mean=mean, covariance_matrix=lazy_covar, interleaved=True)




MultitaskMultivariateNormalType = Annotated[_MultitaskMultivariateNormalType, ...]