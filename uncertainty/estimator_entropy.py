from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import List

from batchbald_redux.batchbald import CandidateBatch, compute_conditional_entropy
from uncertainty.current_batch import CurrentBatch
from uncertainty.rank2 import Rank1Update, Rank1Updates


from toma import toma
from tqdm import tqdm
from typeguard import typechecked
from batchbald_redux import joint_entropy
from utils.typing import TensorType
import torch
import string

# Here we encapsalate the logic for actually estimating the joint entropy

@dataclass
class Sampling:
    batch_samples: int = 0
    per_samples: int = 0
    sum_samples: int = 0


def get_entropy_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, num_samples: int, dtype=None, device=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    if batch_size == 0:
        return CandidateBatch(candidate_scores, candidate_indices)

    batch_joint_entropy = joint_entropy.DynamicJointEntropy(
        num_samples, batch_size - 1, K, C, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

        scores_N[candidate_indices] = -float("inf")

        candidate_score, candidate_index = scores_N.max(dim=0)

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    return CandidateBatch(candidate_scores, candidate_indices)





class MVNJointEntropyEstimator(ABC):
    def __init__(self, batch: CurrentBatch, likelihood, samples: Sampling) -> None:
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

    @abstractmethod
    def get_current_batch(self) -> CurrentBatch:
        pass


class CombinedJointEntropyEstimator(MVNJointEntropyEstimator):
    def __init__(self, batch: CurrentBatch, likelihood, samples: Sampling) -> None:
        super().__init__(batch, likelihood, samples)
        self.count = 0
        self.samples = samples
        self.likelhood = likelihood
        self.inner : MVNJointEntropyEstimator = ExactJointEntropyEstimator(batch, likelihood, samples)

    def compute(self) -> TensorType:
        return self.inner.compute()

    def compute_batch(self, candidates: Rank1Updates) -> TensorType["N"]:
        return self.inner.compute_batch(candidates)

    def add_variable(self, new: Rank1Update) -> None:
        if self.count == 3:
            self.inner = SampledJointEntropyEstimator(self.inner.get_current_batch(), self.likelhood, self.samples)
        self.count += 1
        return self.inner.add_variable(new)

    def get_current_batch(self) -> CurrentBatch:
        return self.inner.get_current_batch()
    
class SampledJointEntropyEstimator(MVNJointEntropyEstimator):
    def __init__(self, batch: CurrentBatch, likelihood, samples: Sampling) -> None:
        self.batch = batch
        self.likelihood = likelihood
        self.sum_samples = samples.sum_samples
        self.batch_samples = samples.batch_samples
        self.per_samples = samples.per_samples
        self.create_samples()
        
        super().__init__(batch, likelihood, samples)


    # We call this function to create the samples for the batch distribution
    # We use r2c and the current batch dist to create the samples and conditional distributions
    def create_samples(self) -> None:
        if self.batch.num_points == 0:
            if torch.cuda.is_available():
                self.probs = torch.ones(self.batch_samples * self.sum_samples, self.batch_samples, device='cuda')
            else:
                self.probs = torch.ones(self.batch_samples * self.sum_samples, self.batch_samples)
            self.likelihood_samples = torch.zeros(self.batch_samples, 0, batch.num_cat)
            return

        distribution = self.batch.distribution
        likelihood = self.likelihood
        batch_samples = self.batch_samples
        sum_samples = self.sum_samples
        

        likelihood_samples: TensorType["S", "D", "C"] = distribution.sample(sample_shape=torch.Size([self.batch_samples]))
        self.likelihood_samples = likelihood_samples

        probs_N_K_C = likelihood(likelihood_samples).probs.permute(1, 0, 2)

        choices_N_K_S = joint_entropy.batch_multi_choices(probs_N_K_C, sum_samples).long()

        expanded_choices_N_1_K_S = choices_N_K_S[:, None, :, :]
        expanded_probs_N_K_1_C = probs_N_K_C[:, :, None, :]

        probs_N_K_K_S = joint_entropy.gather_expand(expanded_probs_N_K_1_C, dim=-1, index=expanded_choices_N_1_K_S)
        # exp sum log seems necessary to avoid 0s?
        probs_K_K_S = torch.exp(torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False))
        samples_K_M = probs_K_K_S.reshape((batch_samples, -1))
        self.probs = samples_K_M.t()
        if torch.cuda.is_available():
            self.probs = self.probs.to('cuda', non_blocking=True)


    def compute(self) -> TensorType:
        p = self.probs
        p = torch.mean(p, dim=0)
        return torch.mean(- torch.log(p))

    @typechecked
    def compute_batch(self, candidates: Rank1Updates) -> TensorType["N"]:   
        p_m_k = self.probs


        N = len(candidates)
                
        P = self.per_samples
        L = self.batch_samples
        
        output = torch.zeros(N, device=p_m_k.device)

        @toma.execute.batch(N*L)
        def compute(batchsize: int):
            candidates.reset()
            conditional_dists = self.batch.create_conditionals_from_rank1s(candidates, self.likelihood_samples, batchsize)
            pbar = tqdm(total=N*L, desc="Sampling Batch", leave=False)
            datapoints_size = max(1, batchsize // L)
            samples_size = min(L, batchsize)
            M = L *  self.sum_samples
            C = self.batch.num_cat
            K = L
            for i, candidate in enumerate(conditional_dists):
                n_start = i*datapoints_size
                n_end = min((i+1)*datapoints_size, N)
                p_b_m_c = torch.zeros(n_end - n_start, M, C, device=p_m_k.device)
                for j, distribution in enumerate(candidate):
                    l_start = j*samples_size
                    l_end = min((j+1)* samples_size, L)
                    sample: TensorType["P", "B", "K", "C"] = distribution.sample(sample_shape=torch.Size([P])).squeeze(-2)

                    likelihood_samples: TensorType["P", "B", "K", "C"] = (self.likelihood(sample).probs)
                    p_c: TensorType["B", "K", "C"] = torch.mean(likelihood_samples, dim=0)

                    batch_p: TensorType["M", "K"] = p_m_k[:, l_start: l_end]

                    batch_p_expanded: TensorType[1, "M", "K"] = batch_p.unsqueeze(0)
                    p_m_c_: TensorType["B", "M", "C"] =  batch_p_expanded @ p_c
                    p_b_m_c += p_m_c_
                    pbar.update((n_end - n_start) * (l_end - l_start))
                
                p_b_m_c /=  K
                p_m: TensorType["M"] = torch.mean(p_m_k, dim=1)
                p_1_m_1 = p_m[None,:,None]
                h: TensorType["B", "L", "M"] = - torch.sum( (p_b_m_c / p_1_m_1) * torch.log(p_b_m_c), dim=(1,2)) / M
                output[n_start:n_end].copy_(h, non_blocking=True)
            pbar.close()
        
        return output.cpu()


    def add_variable(self, new: Rank1Update) -> None:
        self.batch = self.batch.append(new)
        self.create_samples()

    def get_current_batch(self) -> CurrentBatch:
        return self.batch

class ExactJointEntropyEstimator(MVNJointEntropyEstimator):
    def __init__(self, batch: CurrentBatch, likelihood, samples: Sampling) -> None:
        self.batch = batch
        self.likelihood = likelihood
        self.samples = samples.batch_samples
        super().__init__(batch, likelihood, samples)

    @staticmethod
    def _compute(samples: TensorType) -> TensorType:
        D = samples.shape[2]
        t = string.ascii_lowercase[:D]
        s =  ','.join(['yz' + c for c in list(t)]) + '->' + 'yz' + t
        l: TensorType["N","S", "D", "C"] = samples
        j: List[TensorType["N", "S", "C"]] = list(torch.unbind(l, dim=-2))
        # This is where the stupid amount of memory happens
        g: TensorType["N", "S", "expanded" : ...] = torch.einsum(s, *j) # We should have num_points dimensions each of size num_cat
        g: TensorType["N", "S", "E"] = torch.flatten(g, start_dim=-D)
        g: TensorType["N", "E"] = torch.mean(g, dim=-2)
        return torch.sum(-g * torch.log(g), dim=-1)

    @staticmethod
    def _compute_from_batch(batch: CurrentBatch, likelihood, samples: int) -> TensorType:
        return ExactJointEntropyEstimator._compute(likelihood(batch.distribution.sample(sample_shape=torch.Size([samples]))).probs)

    def compute(self) -> TensorType:
        return self._compute_from_batch(self.batch, self.likelihood, self.samples)

    def compute_batch(self, pool: Rank1Updates) -> TensorType["N"]:
        N = len(pool)
        output: TensorType["N"] = torch.zeros(N)
        L = self.samples
        likelihood_samples: TensorType["S", "D", "C"] = self.batch.distribution.sample(sample_shape=torch.Size([L]))

        @toma.execute.batch(N*L)
        def compute(batchsize: int):
            pool.reset()
            conditional_dists = self.batch.create_conditionals_from_rank1s(pool, likelihood_samples, batchsize)
            pbar = tqdm(total=N*L, desc="Exact Batch", leave=False)
            datapoints_size = max(1, batchsize // L)
            samples_size = min(L, batchsize)
            C = self.batch.num_cat
            K = L
            for i, candidate in enumerate(conditional_dists):
                n_start = i*datapoints_size
                n_end = min((i+1)*datapoints_size, N)

                samples = []
                for j, distribution in enumerate(candidate):
                    l_start = j*samples_size
                    l_end = min((j+1)* samples_size, L)
                    sample: TensorType["B", "K", "C"] = distribution.sample(sample_shape=torch.Size([1])).squeeze(-2).squeeze(0)
                    samples.append(sample)
                    pbar.update((n_end - n_start) * (l_end - l_start))
                probs_b_k_c = self.likelihood(torch.cat(samples, dim=1)).probs
                if likelihood_samples.shape[1] > 0:
                    probs_k_d_c = self.likelihood(likelihood_samples).probs
                else:
                    probs_k_d_c = likelihood_samples
                probs_B_K_D_C = torch.cat([probs_b_k_c.unsqueeze(2), probs_k_d_c.unsqueeze(0).expand(n_end-n_start,-1,-1,-1)] , dim = 2)
                output[n_start: n_end].copy_(ExactJointEntropyEstimator._compute(probs_B_K_D_C), non_blocking=True)
            pbar.close()
        
        return output.cpu()

    def add_variable(self, new: Rank1Update) -> None:
        self.batch = self.batch.append(new)

    def get_current_batch(self) -> CurrentBatch:
        return self.batch
