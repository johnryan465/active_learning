


from typing import Iterator

from uncertainty.rank2 import Rank1Update, Rank1Updates
from uncertainty.multivariate_normal import MultitaskMultivariateNormalType
import torch
from utils.typing import TensorType
from dataclasses import dataclass

from typeguard import typechecked

@dataclass
class CurrentBatch:
    distribution: MultitaskMultivariateNormalType
    num_cat: int
    num_points: int = 0
    cuda: bool = False

    def append(self, rank1: Rank1Update) -> "CurrentBatch":
        new_dist = self.distribution
        cross_mat: TensorType["C", 1, "D"] = rank1.cross_covariance
        self_cov: TensorType["C", 1, 1] = rank1.covariance
        new_mean: TensorType[1, "C"] = rank1.mean
        if self.cuda:
            new_mean = new_mean.cuda()
            cross_mat = cross_mat.cuda()
            self_cov = self_cov.cuda()
        # Next we update the current distribution
        _mean = torch.cat([self.distribution.mean, new_mean], dim=0)
        _covar = self.distribution.lazy_covariance_matrix.base_lazy_tensor.cat_rows(cross_mat, self_cov).evaluate()

        new_dist = MultitaskMultivariateNormalType.create(_mean, _covar, self.cuda)
        return CurrentBatch(new_dist, self.num_cat, self.num_points+1, self.cuda)

    @staticmethod
    def empty(num_cat: int, cuda: bool) -> "CurrentBatch":
        return CurrentBatch.create_identity(num_cat, 0, cuda)

    @staticmethod
    def create_identity(num_cat: int, num_points: int, cuda: bool) -> "CurrentBatch":
        covar = torch.eye(num_points)[None,:,:].expand(num_cat, -1, -1)
        mean = torch.zeros(num_points, num_cat)
        distribution = MultitaskMultivariateNormalType.create(mean, covar, cuda)
        return CurrentBatch(distribution, num_cat, num_points=num_points, cuda=cuda)

    def get_inverse(self) -> TensorType["C", "D", "D"]:
        sigma_XX: TensorType["C", "D", "D"] = self.distribution.lazy_covariance_matrix.base_lazy_tensor.evaluate()
        sigma_XX_inv: TensorType["C", "D", "D"] = torch.inverse(sigma_XX)
        return sigma_XX_inv

    def get_mean(self) -> TensorType:
        return torch.transpose(self.distribution.mean, -1, -2)

    # We take in a rank 1 update and a sample from this distribution
    @typechecked
    def create_conditional(self, rank1: Rank1Update, sample: TensorType["D", "C"]) -> MultitaskMultivariateNormalType:
        sigma_YY: TensorType["C", 1, 1] = rank1.covariance
        sigma_YX: TensorType["C", 1, "D"] = rank1.cross_covariance
        sigma_XX_inv: TensorType["C", 1, "D"] = self.get_inverse()
        
        # mean = mu_Y + sigma_YX sigma_XX^{-1} (X - mu_x)
        # We want to expand the batch dist mean and covariance so we can broadcast in numbesumr of samples
        mu_Y: TensorType["C", 1] = torch.transpose(rank1.mean, -1, -2)
        mu_X: TensorType["C", "D"] = self.get_mean()
        X: TensorType["C", "D"] = torch.transpose(sample, -1, -2)

        cuda = torch.cuda.is_available()
        if cuda:
            X = X.cuda()
            mu_Y = mu_Y.cuda()
            mu_X = mu_X.cuda()
            sigma_XX_inv = sigma_XX_inv.cuda()
            sigma_YX = sigma_YX.cuda()

        conditional_cov: TensorType["C", 1, 1] = sigma_YY - (sigma_YX @ sigma_XX_inv @ torch.transpose( sigma_YX, -1, -2))

        tmp_vector: TensorType["C", "D", 1] = (X - mu_X).unsqueeze(-1)

        # We can let pytorch auto broadcast the matrix multiplication
        tmp_matrix: TensorType["C", "D" , 1] = (sigma_YX @ sigma_XX_inv)
        conditional_mean: TensorType["C", 1, 1] = (tmp_matrix @ tmp_vector)
        conditional_mean = mu_Y + conditional_mean.squeeze(-1)
        
        return MultitaskMultivariateNormalType.create(conditional_mean, conditional_cov, cuda=cuda)
    
    def _create_conditionals_from_rank1s_util(self, rank1s: Rank1Updates, samples: TensorType["L", "D", "C"]) -> MultitaskMultivariateNormalType:
        L = samples.shape[0]
        rank1s.reset()

        means = []
        covariance = []
        cross_covariance = []
        for rank1 in rank1s:
            means.append(rank1.mean)
            covariance.append(rank1.covariance)
            cross_covariance.append(rank1.cross_covariance)
        sigma_YY: TensorType["N", "C", 1, 1] = torch.stack(covariance, dim=0)
        sigma_YX: TensorType["N", "C", 1, "D"] = torch.stack(cross_covariance, dim=0)
        sigma_XX_inv: TensorType["C", 1, "D"] = self.get_inverse()

        
        # mean = mu_Y + sigma_YX sigma_XX^{-1} (X - mu_x)
        # We want to expand the batch dist mean and covariance so we can broadcast in numbesumr of samples
        mu_Y: TensorType["N", "C", 1] = torch.transpose(torch.stack(means, dim=0), -1, -2)
        mu_X: TensorType["C", "D"] = self.get_mean()
        X: TensorType["L", "C", "D"] = torch.transpose(samples, -1, -2)

        if self.cuda:
            X = X.cuda()
            mu_Y = mu_Y.cuda()
            mu_X = mu_X.cuda()
            sigma_XX_inv = sigma_XX_inv.cuda()
            sigma_YX = sigma_YX.cuda()

        conditional_cov: TensorType["N", "C", 1, 1] = sigma_YY - (sigma_YX @ sigma_XX_inv @ torch.transpose( sigma_YX, -1, -2))

        tmp_vector: TensorType[1, "L", "C", "D", 1] = (X - mu_X).unsqueeze(-1).unsqueeze(0)

        # We can let pytorch auto broadcast the matrix multiplication
        tmp_matrix: TensorType["N", 1, "C", "D" , 1] = (sigma_YX @ sigma_XX_inv).unsqueeze(1)
        conditional_mean: TensorType["N", "L", "C", 1, 1] = (tmp_matrix @ tmp_vector)
        conditional_mean = (mu_Y.unsqueeze(1) + conditional_mean.squeeze(-1)).squeeze(-1).unsqueeze(-2)

        conditional_cov: TensorType["N", "L", "C", 1, 1] = conditional_cov.unsqueeze(1).expand(-1, L,-1,-1,-1)
        return MultitaskMultivariateNormalType.create(conditional_mean, conditional_cov, self.cuda)


    # We take in a rank 1 update and a sample from this distribution
    def create_conditionals_from_rank1s(self, rank1s: Rank1Updates, samples: TensorType["L", "D", "C"], size: int) -> Iterator[Iterator[MultitaskMultivariateNormalType]]:
        L = samples.shape[0]
        datapoints_size = max(size // L, 1)
        samples_size = min(size, L)
        rank1s_count = 0
        max_rank1s_count = len(rank1s)
        datapoints_size = min(datapoints_size, max_rank1s_count)

        while rank1s_count < max_rank1s_count:
            l = [x for _, x in zip(range(datapoints_size), rank1s)]
            updates = Rank1Updates(already_computed=l)
            rank1s_count += len(l)
            yield (self._create_conditionals_from_rank1s_util(updates, samples[idx:  min(idx+samples_size, L)]) for idx in range(0, L, samples_size))

    def __eq__(self, other):
        if isinstance(other, CurrentBatch):
            return (self.num_points == other.num_points) and (self.num_cat == other.num_cat) and (self.distribution == other.distribution)
        return False
