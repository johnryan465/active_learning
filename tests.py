from torch import cuda
from uncertainty.multivariate_normal import MultitaskMultivariateNormalType
from gpytorch.lazy.block_interleaved_lazy_tensor import BlockInterleavedLazyTensor
from gpytorch.lazy.non_lazy_tensor import lazify
import torch
from uncertainty.rank2 import Rank1Update
from uncertainty.estimator_entropy import CurrentBatch
import unittest

class CurrentBatchTest(unittest.TestCase):
    def test_empty_append(self):
        batch = CurrentBatch.create_identity(1, 1, False)

        empty_batch = CurrentBatch.empty(1, cuda=False)
        update = Rank1Update(mean=torch.zeros(1,1), covariance=torch.ones(1,1,1), cross_covariance=torch.zeros(1,1,0))
        new_batch = empty_batch.append(update)
        self.assertEqual(batch, new_batch)

    def test_basic_conditional(self):
        batch = CurrentBatch.create_identity(1, 1, False)

        update = Rank1Update(mean=torch.zeros(1,1), covariance=torch.ones(1,1,1), cross_covariance=torch.zeros(1,1,1))
        new_dist = batch.create_conditional(update, torch.zeros(1, 1))
        self.assertEqual(batch.distribution, new_dist)
    
    def test_basic_ones_conditional(self):
        batch = CurrentBatch.create_identity(1, 1, False)
        update = Rank1Update(mean=torch.zeros(1,1), covariance=torch.ones(1,1,1), cross_covariance=0.5*torch.ones(1,1,1))
        new_dist = batch.create_conditional(update, torch.ones(1, 1))

        correct_dist = MultitaskMultivariateNormalType(torch.tensor([[0.5]]), torch.tensor([[0.75]]))
        self.assertEqual(correct_dist, new_dist)


if __name__ == '__main__':
    unittest.main()