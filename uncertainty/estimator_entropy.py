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

        if torch.cuda.is_available():
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
        conditional_cov = BlockInterleavedLazyTensor(lazify(conditional_cov))
        return MultitaskMultivariateNormal(mean=conditional_mean, covariance_matrix=conditional_cov)


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
        self.samples_sum = 60
        self.batch_samples = 300
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


        p: TensorType["S", "S", "D", "E"] = torch.log(p)
        p: TensorType["S", "S", "E"] = torch.sum(p, dim=2) # For each of the samples we have a random sample of log probs
        p: TensorType["S", "S*E"] = torch.flatten(p, start_dim=1)

        self.log_probs = p

    def compute(self) -> TensorType:
        p = self.log_probs.exp()
        p = torch.mean(p, dim=0)
        return torch.mean(- torch.log(p))

    @typechecked
    def _compute_batch(self, candidates: Rank1Updates) -> TensorType["N"]:   
        log_p = self.log_probs

        if torch.cuda.is_available():
            log_p = log_p.cuda()


        N = candidates.size
        
        P = self.per_samples
        L = self.batch_samples
        X = L * self.samples_sum


        output = torch.zeros(N)
        # We are going to compute the entropy in stages, the entropy of the current batch plus the entropy of the candiate
        # conditioned on the batch.
        # Sample from current batch create conditional, sample from conditional
        @toma.execute.batch(N*L)
        def compute(batchsize: int):
            candidates.reset()
            batch_samples: TensorType["L", "D", "Y"] = self.likelihood_samples
            for i, candidate in enumerate(candidates):
                # p_l_d_c = torch.zeros(L, self.batch.num_points + 1, 10)
                sample_list = []
                candidates_ = Rank1Updates(already_computed=[candidate])
                conditional_dists = next(self.batch.create_conditionals_from_rank1s(candidates_, batch_samples, batchsize))
                for cond_dist in conditional_dists:
                    point_sample: TensorType["P", "L", 1, "Y"] = cond_dist.sample(sample_shape=torch.Size([P])).squeeze(1)
                    # combined_sample = torch.cat([batch_samples, point_sample], dim=1)
                    point_sample = self.likelihood(point_sample).probs
                    # point_sample = torch.transpose(point_sample, 0, 1)
                    point_sample = torch.mean(point_sample, dim=0)
                    sample_list.append(point_sample)
                p_l_1_y: TensorType["L", 1, "Y"]  = torch.cat(sample_list, dim=0)
                p_l_y = p_l_1_y.squeeze(1)
                p_l_x: TensorType["L", "X"] = log_p.exp()
                p_l_x_1: TensorType["L", "X", 1] = p_l_x[:,:,None]
                p_l_x_y: TensorType["L", "X", "Y"] = p_l_x_1 * p_l_1_y
                p_x = torch.mean(p_l_x, dim= 0)
                p_x_y = torch.mean(p_l_x_y, dim=0)

                h = torch.sum((p_l_1_y * (p_l_x_1 / p_x[None,:,None])) * torch.log(p_x_y[None,:,:]), dim=2)
                output[i] = -torch.mean(h)
        print(output)
        return output
    
    @typechecked
    def compute_batch(self, candidates: Rank1Updates) -> TensorType["N"]:   
        log_p = self.log_probs

        if torch.cuda.is_available():
            p_l_x = log_p.exp().cuda()
        else:
            p_l_x = log_p.exp()


        N = candidates.size
        
        P = self.per_samples
        L = self.batch_samples


        output = torch.zeros(N)
        # We are going to compute the entropy in stages, the entropy of the current batch plus the entropy of the candiate
        # conditioned on the batch.

        @toma.execute.batch(N*L)
        def compute(batchsize: int):
            candidates.reset()
            conditional_dists = self.batch.create_conditionals_from_rank1s(candidates, self.likelihood_samples, batchsize)
            pbar = tqdm(total=N*L, desc="Sampling Batch", leave=False)
            datapoints_size = max(1, batchsize // L)
            samples_size = min(L, batchsize)
            X = L *  self.samples_sum
            Y = self.batch.num_cat
            # p_n_l_1_y = torch.zeros(datapoints_size, L, 1, Y)
            for i, candidate in enumerate(conditional_dists):
                n_start = i*datapoints_size
                n_end = min((i+1)*datapoints_size, N)
                p_n_l_1_y = torch.zeros(n_end - n_start, L, 1, Y, device=p_l_x.device)
                p_n_x_y = torch.zeros(n_end - n_start, X, Y, device=p_l_x.device)
                for j, distribution in enumerate(candidate):
                    l_start = j*samples_size
                    l_end = min((j+1)* samples_size, L)
                    sample: TensorType["P", "N", "L", "Y"] = distribution.sample(sample_shape=torch.Size([P])).squeeze(-2)

                    likelihood_samples: TensorType["P", "N", "L", "Y"] = (self.likelihood(sample).probs)
                    p_y: TensorType["N", "L", "Y"] = torch.mean(likelihood_samples, dim=0)

                    batch_p: TensorType["L", "X"] = p_l_x[l_start: l_end,]
                    p_expanded: TensorType["N", "L", 1, "Y"] = p_y.unsqueeze(-2)
                    p_n_l_1_y[:,l_start:l_end,:,:].copy_(p_expanded)
                    batch_p_expanded: TensorType[1, "L", "X", 1] = batch_p.unsqueeze(-1).unsqueeze(0)
                    
                    p_x_y_: TensorType["N", "L", "X", "Y"] =  batch_p_expanded * p_expanded # p(x,y | l)
                    p_x_y_: TensorType["N", "X", "Y"] =  torch.sum(p_x_y_, dim=1)
                    if j == 0:
                        p_n_x_y = p_x_y_
                    else:
                        p_n_x_y += p_x_y_
                    pbar.update((n_end - n_start) * (l_end - l_start))

                p_n_x_y: TensorType["N", "X", "Y"] = p_n_x_y / L
                p_x: TensorType["X"] = torch.mean(p_l_x, dim=0)
                p_n_l_x_1: TensorType["N", "L", "X", 1] = p_l_x[None,:,:,None].expand(n_end - n_start, -1,-1,-1)
                h: TensorType["N", "L", "X"] = torch.sum((p_n_l_1_y * (p_n_l_x_1 / p_x[None,None,:,None])) * torch.log(p_n_x_y[:,None,:,:]), dim=3)
                p: TensorType["N"] = -torch.mean(h, dim = (1,2)) # H(Y | X)
                del p_n_l_1_y
                del p_n_x_y
                output[n_start:n_end].copy_(p, non_blocking=True)
            pbar.close()
        
        batch_entropy: TensorType = self.compute().cpu()
        print(batch_entropy)
        conditioned_entropy = output.cpu()
        print(conditioned_entropy)
        conditioned_entropy = conditioned_entropy # + batch_entropy
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
    def _compute(samples: TensorType["S","D", "C"]) -> TensorType:
        D = samples.shape[1]
        t = string.ascii_lowercase[:D]
        s =  ','.join(['z' + c for c in list(t)]) + '->' + 'z' + t
        l: TensorType["S", "D", "C"] = samples
        j: List[TensorType["S", "C"]] = list(torch.unbind(l, dim=-2))
        # This is where the stupid amount of memory happens
        g: TensorType["S", "expanded" : ...] = torch.einsum(s, *j) # We should have num_points dimensions each of size num_cat
        g: TensorType["S", "E"] = torch.flatten(g, start_dim=1)
        g: TensorType["E"] = torch.mean(g, dim=0)
        return torch.sum(-g * torch.log(g), dim=0)

    @staticmethod
    def _compute_from_batch(batch: CurrentBatch, likelihood, samples: int) -> TensorType:
        return ExactJointEntropyEstimator._compute(likelihood(batch.distribution.sample(sample_shape=torch.Size([samples]))).probs)

    def compute(self) -> TensorType:
        return self._compute_from_batch(self.batch, self.likelihood, self.samples)

    def compute_batch(self, pool: Rank1Updates) -> TensorType["N"]:
        N = len(pool)
        output: TensorType["N"] = torch.zeros(N)
        pbar = tqdm(total=N, desc="Exact Batch", leave=False)
        for i, candidate in enumerate(pool):
            possible_batch = self.batch.append(candidate)
            output[i] = ExactJointEntropyEstimator._compute_from_batch(possible_batch, self.likelihood, self.samples)
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
        log_probs_N_K_C: TensorType["datapoints", "samples", "num_cat"] = ((likelihood(batch.distribution.sample(sample_shape=torch.Size([1000]))).logits)).permute(1,0,2) # type: ignore
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

# p: TensorType["S", "S", "D", "E"] = torch.gather(l, 3, choices)


# p: TensorType["S", "S", "D", "E"] = torch.log(p)
# p: TensorType["S", "S", "E"] = torch.sum(p, dim=2) # For each of the samples we have a random sample of log probs
# p: TensorType["S", "S*E"] = torch.flatten(p, start_dim=1)

# self.log_probs = p
# p = self.log_probs.exp()
# p = torch.mean(p, dim=0)
# torch.mean(- torch.log(p))
