from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Iterator, List
from uncertainty.rank2 import Rank1Update, Rank1Updates, Rank2Combine
from gpytorch.distributions import distribution

from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from gpytorch.lazy.block_interleaved_lazy_tensor import BlockInterleavedLazyTensor
from gpytorch.lazy.non_lazy_tensor import lazify
from toma import toma
from tqdm import tqdm
from typeguard import typechecked
from batchbald_redux import joint_entropy
from utils.typing import MultitaskMultivariateNormalType, TensorType

import torch
import string

# Here we encapsalate the logic for actually estimating the joint entropy




@dataclass
class CurrentBatch:
    distribution: MultitaskMultivariateNormalType
    num_cat: int
    num_points: int = 0

    def append(self, rank1: Rank1Update) -> "CurrentBatch":
        new_dist = self.distribution
        cross_mat: TensorType["C", 1, "D"] = rank1.cross_covariance
        self_cov: TensorType["C", 1, 1] = rank1.covariance
        new_mean: TensorType[1, "C"] = rank1.mean
        if torch.cuda.is_available():
            new_mean = new_mean.cuda()
            cross_mat = cross_mat.cuda()
            self_cov = self_cov.cuda()
        # Next we update the current distribution
        _mean = torch.cat( [self.distribution.mean, new_mean], dim=0)
        _covar = self.distribution.lazy_covariance_matrix.base_lazy_tensor.cat_rows(cross_mat, self_cov).evaluate()
        if torch.cuda.is_available():
            _mean = _mean.cuda()
            _covar = BlockInterleavedLazyTensor(lazify(_covar).cuda(), block_dim=-3)
        else:
            _covar = BlockInterleavedLazyTensor(lazify(_covar), block_dim=-3)

        new_dist = MultitaskMultivariateNormal(mean=_mean, covariance_matrix=_covar)
        return CurrentBatch(new_dist, self.num_cat, self.num_points+1)

    @staticmethod
    def empty(num_cat: int) -> "CurrentBatch":
        covar = torch.eye(0)[None,:,:].expand(num_cat, -1, -1)
        mean = torch.zeros(0, num_cat)
        if torch.cuda.is_available():
            covar = BlockInterleavedLazyTensor(lazify(covar.cuda()).cuda())
            mean = mean.cuda()
        else:
            covar = BlockInterleavedLazyTensor(lazify(covar))
        
        distribution = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covar)
        return CurrentBatch(distribution, num_cat, num_points=0)

    def get_inverse(self) -> TensorType:
        sigma_XX: TensorType["C", "D", "D"] = self.distribution.lazy_covariance_matrix.base_lazy_tensor.evaluate()
        sigma_XX_inv: TensorType["C", "D", "D"] = torch.inverse(sigma_XX)
        return sigma_XX_inv

    def get_mean(self) -> TensorType:
        return torch.transpose(self.distribution.mean, -1, -2)

    # We take in a rank 1 update and a sample from this distribution
    def create_conditional(self, rank1: Rank1Update, sample: TensorType["D", "C"]) -> MultitaskMultivariateNormalType:
        sigma_YY: TensorType["C", 1, 1] = rank1.covariance
        sigma_YX: TensorType["C", 1, "D"] = rank1.cross_covariance
        sigma_XX_inv: TensorType["C", 1, "D"] = self.get_inverse()
        
        # mean = mu_Y + sigma_YX sigma_XX^{-1} (X - mu_x)
        # We want to expand the batch dist mean and covariance so we can broadcast in numbesumr of samples
        mu_Y: TensorType["C", 1] = torch.transpose(rank1.mean, -1, -2)
        mu_X: TensorType["C", "D"] = self.get_mean()
        X: TensorType["C", "D"] = torch.transpose(sample, -1, -2)

        if torch.cuda.is_available():
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
        
        conditional_cov = BlockInterleavedLazyTensor(lazify(conditional_cov))
        return MultitaskMultivariateNormal(mean=conditional_mean, covariance_matrix=conditional_cov)


class MVNJointEntropyEstimator(ABC):
    def __init__(self, batch: CurrentBatch, likelihood, samples: int) -> None:
        pass

    @abstractmethod
    def compute(self) -> TensorType:
        pass

    @abstractmethod
    def compute_batch(self, candidates: Rank1Updates) -> TensorType["N"]:
        pass

    @abstractmethod
    def add_variable(self, new: Rank1Update) -> None:
        pass


class SampledJointEntropyEstimator(MVNJointEntropyEstimator):
    def __init__(self, batch: CurrentBatch, likelihood, samples: int) -> None:
        self.batch = batch
        self.likelihood = likelihood
        self.samples_sum = 10
        self.batch_samples = 3000
        self.per_samples = 100
        self.log_probs = torch.zeros(self.batch_samples, self.batch_samples * self.samples_sum)

        self.likelihood_samples = torch.zeros(self.batch_samples, 0, batch.num_cat)
        super().__init__(batch, likelihood, samples)


    # We call this function to create the samples for the batch distribution
    # We use r2c and the current batch dist to create the samples and conditional distributions
    def create_samples(self) -> None:
        distribution = self.batch.distribution
        likelihood = self.likelihood
        batch_samples = self.batch_samples
        sum_samples = self.samples_sum

        likelihood_samples: TensorType["S", "D", "C"] = likelihood(distribution.sample(sample_shape=torch.Size([self.batch_samples]))).probs

        l_shape = likelihood_samples.shape
        likelihood_samples: TensorType["S * D", "C"] = likelihood_samples.reshape((-1, l_shape[-1]))
        # Instead of using einsum we will sample from the possible 
        # indexes, we wish to keep the same for each datapoints samples

        
        choices: TensorType["S * D", "E"] = torch.multinomial(likelihood_samples, sum_samples, replacement=True)
        choices: TensorType["S", "D", "E"] = choices.reshape( list(l_shape[:-1]) + [-1])
        likelihood_samples: TensorType["S", "D", "C"] = likelihood_samples.reshape(l_shape)
        
        self.likelihood_samples = likelihood_samples


        l: TensorType["S", "S", "D", "C"] = likelihood_samples[:,None,:,:].expand(-1, batch_samples, -1, -1)
        choices: TensorType["S", "S", "D", "E"] = choices[None,:,:,:].expand(batch_samples, -1, -1, -1)
        p: TensorType["S", "S", "D", "E"] = torch.gather(l, 3, choices)


        p: TensorType["S", "S", "D", "E"] = torch.log(p, out=p)
        p: TensorType["S", "S", "E"] = torch.sum(p, dim=2) # For each of the samples we have a random sample of log probs
        p: TensorType["S", "S*E"] = torch.flatten(p, start_dim=1)

        self.log_probs = p

    def compute(self) -> TensorType:
        p = self.log_probs.exp()
        p = torch.mean(p, dim=0)
        return torch.mean(- torch.log(p))

    @typechecked
    def compute_batch(self, candidates: Rank1Updates) -> TensorType["N"]:
        # We can exactly compute a larger sized exact distribution
        # As the task batches are independent we can chunk them
        # If we haven't added any variables yet, the coditional doesn't exist yet, but we have
        # nothing to conditon on. We have only 1 "sample", which has a probability of 1            
        log_p = self.log_probs

        if torch.cuda.is_available():
            log_p = log_p.cuda()


        N = candidates.size
        
        P = self.per_samples

        output = torch.zeros(N)
        # We are going to compute the entropy in stages, the entropy of the current batch plus the entropy of the candiate
        # conditioned on the batch.

        @toma.execute.batch(N)
        def compute(batchsize: int):
            pbar = tqdm(total=N, desc="Sampling Batch", leave=False)
            L = self.batch_samples
            X = L *  self.samples_sum
            Y = self.batch.num_cat
            
            p_l_x = log_p.exp()
            for i, candidate in enumerate(candidates):
                p_x_y: TensorType["X", "Y"] = torch.zeros(X, Y)
                for j in range(self.batch_samples):
                    pool_rank1 = candidate
                    sample = self.likelihood_samples[j]
                    distribution = self.batch.create_conditional(pool_rank1, sample)


                    # We have L samples from the batch distribution
                    # We have B samples from the sum of the batch distribution
                    # We have P samples from the conditional distribution
                    # We have N points on the distribution
                    # We have E samples from the sum of the conditional
                    sample: TensorType["P", "Y", 1] = distribution.sample(sample_shape=torch.Size([P]))

                    # print(sample)
                    likelihood_samples: TensorType["P", "Y"] = (self.likelihood(sample).probs).squeeze(-2)
                    p_y: TensorType["Y"] = torch.mean(likelihood_samples, dim=0)

                    batch_p: TensorType["X"] = p_l_x[j]
                    p_expanded: TensorType[1, "Y"] = p_y.unsqueeze(-2)
                    batch_p_expanded: TensorType["X", 1] = batch_p.unsqueeze(-1)
                    
                    p_x_y_: TensorType["X", "Y"] =  batch_p_expanded * p_expanded # p(x,y | l)
                    p_x_y += p_x_y_.cpu()
                # print(p_l_x_y)
                p_x_y = p_x_y / L
                p_x = torch.mean(p_l_x, dim=0).cpu()
                p_y_given_x: TensorType["X", "Y"] = p_x_y / (p_x[:,None])
                p: TensorType["X", "Y"] = - torch.log(p_y_given_x)  * p_y_given_x # - p(y | x) log p(y | x)
                p: TensorType["X"] = torch.sum(p, 1)  # H(Y | X=x)
                p: TensorType = torch.mean(p, dim = 0) # H(Y | X)
                output[i] = p
                pbar.update(1)

            pbar.close()
        
        batch_entropy: TensorType = self.compute().cpu()
        conditioned_entropy = output.cpu()
        conditioned_entropy = conditioned_entropy + batch_entropy
        return conditioned_entropy

    def add_variable(self, new: Rank1Update) -> None:
        self.batch = self.batch.append(new)
        self.create_samples()



class ExactJointEntropyEstimator(MVNJointEntropyEstimator):
    def __init__(self, batch: CurrentBatch, likelihood, samples: int) -> None:
        self.batch = batch
        self.likelihood = likelihood
        self.samples = samples
        super().__init__(batch, likelihood, samples)

    @staticmethod
    def _compute(batch: CurrentBatch, likelihood, samples: int) -> TensorType:
        D = batch.num_points
        t = string.ascii_lowercase[:D]
        s =  ','.join(['z' + c for c in list(t)]) + '->' + 'z' + t
        l: TensorType["S", "D", "C"] = likelihood(batch.distribution.sample(sample_shape=torch.Size([samples]))).probs
        j: List[TensorType["S", "C"]] = list(torch.unbind(l, dim=-2))
        # This is where the stupid amount of memory happens
        g: TensorType["S", "expanded" : ...] = torch.einsum(s, *j) # We should have num_points dimensions each of size num_cat
        g: TensorType["S", "E"] = torch.flatten(g, start_dim=1)
        g: TensorType["E"] = torch.mean(g, dim=0)
        return torch.sum(-g * torch.log(g), dim=0)

    def compute(self) -> TensorType:
        return self._compute(self.batch, self.likelihood, self.samples)

    def compute_batch(self, pool: Rank1Updates) -> TensorType["N"]:
        N = len(pool)
        output: TensorType["N"] = torch.zeros(N)
        pbar = tqdm(total=N, desc="Exact Batch", leave=False)
        for i, candidate in enumerate(pool):
            possible_batch = self.batch.append(candidate)
            output[i] = ExactJointEntropyEstimator._compute(possible_batch, self.likelihood, self.samples)
            pbar.update(1)
        pbar.close()
        return output

    def add_variable(self, new: Rank1Update) -> None:
        self.batch = self.batch.append(new)


class BBReduxJointEntropyEstimator(MVNJointEntropyEstimator):
    def __init__(self, batch: CurrentBatch, likelihood, samples: int) -> None:
        self.batch = batch
        self.likelihood = likelihood
        self.samples = samples
        super().__init__(batch, likelihood, samples)

    @staticmethod
    def _compute(batch: CurrentBatch, likelihood, samples: int) -> TensorType:
        log_probs_N_K_C: TensorType["datapoints", "samples", "num_cat"] = ((likelihood(batch.distribution.sample(sample_shape=torch.Size([2000]))).logits)).permute(1,0,2) # type: ignore
        N, K, C = log_probs_N_K_C.shape

        batch_joint_entropy = joint_entropy.DynamicJointEntropy(
            samples, N, K, C
        )

        batch_joint_entropy.add_variables(log_probs_N_K_C)

        return batch_joint_entropy.compute()

    def compute(self) -> TensorType:
        return self._compute(self.batch, self.likelihood, self.samples)

    def compute_batch(self, pool: Rank1Updates) -> TensorType["N"]:
        N = len(pool)
        output: TensorType["N"] = torch.zeros(N)
        pbar = tqdm(total=N, desc="BBRedux Batch", leave=False)
        for i, candidate in enumerate(pool):
            possible_batch = self.batch.append(candidate)
            output[i] = BBReduxJointEntropyEstimator._compute(possible_batch, self.likelihood, self.samples)
            pbar.update(1)
        pbar.close()
        return output

    def add_variable(self, new: Rank1Update) -> None:
        self.batch = self.batch.append(new)