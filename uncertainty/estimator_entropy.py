from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import List
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
    samples_sum: int = 0

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


class SampledJointEntropyEstimator(MVNJointEntropyEstimator):
    def __init__(self, batch: CurrentBatch, likelihood, samples: Sampling) -> None:
        self.batch = batch
        self.likelihood = likelihood
        self.samples_sum = samples.samples_sum
        self.batch_samples = samples.batch_samples
        self.per_samples = samples.per_samples
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
        self.likelihood_samples = likelihood_samples

        probs_N_K_C = likelihood_samples.permute(1, 0, 2)

        choices_N_K_S = joint_entropy.batch_multi_choices(probs_N_K_C, sum_samples).long()

        expanded_choices_N_1_K_S = choices_N_K_S[:, None, :, :]
        expanded_probs_N_K_1_C = probs_N_K_C[:, :, None, :]

        probs_N_K_K_S = joint_entropy.gather_expand(expanded_probs_N_K_1_C, dim=-1, index=expanded_choices_N_1_K_S)
        # exp sum log seems necessary to avoid 0s?
        log_probs_K_K_S = torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False)
        samples_K_M = log_probs_K_K_S.reshape((batch_samples, -1))

        self.log_probs = samples_K_M

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
                sample_list = []
                candidates_ = Rank1Updates(already_computed=[candidate])
                conditional_dists = next(self.batch.create_conditionals_from_rank1s(candidates_, batch_samples, batchsize))
                for cond_dist in conditional_dists:
                    point_sample: TensorType["P", "L", 1, "Y"] = cond_dist.sample(sample_shape=torch.Size([P])).squeeze(1)
                    point_sample = self.likelihood(point_sample).probs
                    point_sample = torch.mean(point_sample, dim=0)
                    sample_list.append(point_sample)
                p_l_1_y: TensorType["L", 1, "Y"]  = torch.cat(sample_list, dim=0)
                p_l_x: TensorType["L", "X"] = log_p.exp()
                p_l_x_1: TensorType["L", "X", 1] = p_l_x[:,:,None]
                p_l_x_y: TensorType["L", "X", "Y"] = p_l_x_1 @ p_l_1_y
                p_x: TensorType["X"] = torch.mean(p_l_x, dim=0)
                p_x_y: TensorType["X", "Y"] = torch.mean(p_l_x_y, dim=0)
                e: TensorType["X", "Y"] = p_x_y
                y: TensorType["X", "Y"] = (e / p_x[:, None]) * torch.log(p_x_y)
                h: TensorType["X"] = torch.sum(y, dim=1)
                output[i] = - torch.mean(h)

        return output
    
    @typechecked
    def compute_batch(self, candidates: Rank1Updates) -> TensorType["N"]:   
        log_p = self.log_probs

        if torch.cuda.is_available():
            p_k_m = log_p.exp().cuda()
        else:
            p_k_m = log_p.exp()


        N = candidates.size
        
        P = self.per_samples
        L = self.batch_samples


        output = torch.zeros(N)
        # We are going to compute the entropy in stages, the entropy of the current batch plus the entropy of the candiate
        # conditioned on the batch/s]

        # Dimension annotations
        # N - per data point
        # L - samples from the underlying MVN
        # X - samples from p(x)
        # Y - a dimension of size C for the candidate points
        # P - The number of samples we use to estimate p(y|l)

        @toma.execute.batch(N*L)
        def compute(batchsize: int):
            candidates.reset()
            conditional_dists = self.batch.create_conditionals_from_rank1s(candidates, self.likelihood_samples, batchsize)
            pbar = tqdm(total=N*L, desc="Sampling Batch", leave=False)
            datapoints_size = max(1, batchsize // L)
            samples_size = min(L, batchsize)
            M = L *  self.samples_sum
            C = self.batch.num_cat
            K = L
            # p_n_l_1_y = torch.zeros(datapoints_size, L, 1, Y)
            for i, candidate in enumerate(conditional_dists):
                n_start = i*datapoints_size
                n_end = min((i+1)*datapoints_size, N)
                p_b_m_c = torch.zeros(n_end - n_start, M, C, device=p_k_m.device)
                for j, distribution in enumerate(candidate):
                    l_start = j*samples_size
                    l_end = min((j+1)* samples_size, L)
                    sample: TensorType["P", "B", "K", "C"] = distribution.sample(sample_shape=torch.Size([P])).squeeze(-2)

                    likelihood_samples: TensorType["P", "B", "K", "C"] = (self.likelihood(sample).probs)
                    p_c: TensorType["B", "K", "C"] = torch.mean(likelihood_samples, dim=0)

                    batch_p: TensorType["K", "M"] = p_k_m[l_start: l_end,]

                    batch_p_expanded: TensorType["B", "M", "K"] = batch_p.unsqueeze(0).expand(n_end - n_start, -1, -1).permute(0, 2, 1)
                    p_m_c_: TensorType["B", "M", "C"] =  batch_p_expanded @ p_c
                    if j == 0:
                        p_b_m_c = p_m_c_
                    else:
                        p_b_m_c += p_m_c_
                    pbar.update((n_end - n_start) * (l_end - l_start))
                
                p_b_m_c /=  K
                p_m: TensorType["M"] = torch.mean(p_k_m, dim=0)
                p_1_m_1 = p_m[None,:,None]
                h: TensorType["B", "L", "M"] = - torch.sum( (p_b_m_c / p_1_m_1) * torch.log(p_b_m_c), dim=(1,2)) / M
                output[n_start:n_end].copy_(h, non_blocking=True)
            pbar.close()
        
        entropy = output.cpu()
        return entropy

    def add_variable(self, new: Rank1Update) -> None:
        self.batch = self.batch.append(new)
        self.create_samples()

class _SampledJointEntropy:
    """Random variables (all with the same # of categories $C$) can be added via `SampledJointEntropy.add_variables`.

    `SampledJointEntropy.compute` computes the joint entropy.

    `SampledJointEntropy.compute_batch` computes the joint entropy of the added variables with each of the variables in the provided batch probabilities in turn."""

    sampled_joint_probs_M_K: torch.Tensor

    def __init__(self, sampled_joint_probs_M_K: torch.Tensor, likelihood):
        self.sampled_joint_probs_M_K = sampled_joint_probs_M_K
        self.likelihood = likelihood

    @staticmethod
    def empty(K: int, likelihood, device=None, dtype=None) -> "_SampledJointEntropy":
        return _SampledJointEntropy(torch.ones((1, K), device=device, dtype=dtype), likelihood)

    @staticmethod
    def sample(samples: torch.Tensor, M: int, likelihood) -> "_SampledJointEntropy":
        probs_N_K_C = likelihood(samples).probs
        K = probs_N_K_C.shape[1]

        # S: num of samples per w
        S = M // K

        choices_N_K_S = joint_entropy.batch_multi_choices(probs_N_K_C, S).long()

        expanded_choices_N_1_K_S = choices_N_K_S[:, None, :, :]
        expanded_probs_N_K_1_C = probs_N_K_C[:, :, None, :]

        probs_N_K_K_S = joint_entropy.gather_expand(expanded_probs_N_K_1_C, dim=-1, index=expanded_choices_N_1_K_S)
        # exp sum log seems necessary to avoid 0s?
        probs_K_K_S = torch.exp(torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False))
        samples_K_M = probs_K_K_S.reshape((K, -1))

        samples_M_K = samples_K_M.t()
        return _SampledJointEntropy(samples_M_K, likelihood)

    def compute(self) -> torch.Tensor:
        sampled_joint_probs_M = torch.mean(self.sampled_joint_probs_M_K, dim=1, keepdim=False)
        nats_M = -torch.log(sampled_joint_probs_M)
        entropy = torch.mean(nats_M)
        return entropy

    def add_variables(self, log_probs_N_K_C: torch.Tensor, M2: int) -> "_SampledJointEntropy":
        assert self.sampled_joint_probs_M_K.shape[1] == log_probs_N_K_C.shape[1]

        sample_K_M1_1 = self.sampled_joint_probs_M_K.t()[:, :, None]

        new_sample_M2_K = self.sample(log_probs_N_K_C.exp(), M2, self.likelihood).sampled_joint_probs_M_K
        new_sample_K_1_M2 = new_sample_M2_K.t()[:, None, :]

        merged_sample_K_M1_M2 = sample_K_M1_1 * new_sample_K_1_M2
        merged_sample_K_M = merged_sample_K_M1_M2.reshape((K, -1))

        self.sampled_joint_probs_M_K = merged_sample_K_M.t()

        return self

    def compute_batch(self, pool: Rank1Updates, batch, samples, function_samples, per_samples, output_entropies_B=None):
        candidate_samples = []
        for i, candidate in enumerate(pool):
            cond_dists = batch.create_conditionals_from_rank1s(Rank1Updates(already_computed=[candidate]), samples, function_samples)
            dist = next(next(cond_dists))
            sample_ = dist.sample(sample_shape=torch.Size([per_samples])).squeeze(3).squeeze(1)
            sample_ = torch.mean(sample_, dim=0, keepdim=True)
            candidate_samples.append(sample_)

        candidate_samples = torch.cat(candidate_samples, dim=0)
        
        log_probs_B_K_C = self.likelihood(candidate_samples).logits
        assert self.sampled_joint_probs_M_K.shape[1] == log_probs_B_K_C.shape[1]

        B, K, C = log_probs_B_K_C.shape
        M = self.sampled_joint_probs_M_K.shape[0]

        if output_entropies_B is None:
            output_entropies_B = torch.empty(B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device)

        pbar = tqdm(total=B, desc="SampledJointEntropy.compute_batch", leave=False)

        @toma.execute.chunked(log_probs_B_K_C, initial_step=1024, dimension=0)
        def chunked_joint_entropy(chunked_log_probs_b_K_C: torch.Tensor, start: int, end: int):
            b = chunked_log_probs_b_K_C.shape[0]

            probs_b_M_C = torch.empty(
                (b, M, C),
                dtype=self.sampled_joint_probs_M_K.dtype,
                device=self.sampled_joint_probs_M_K.device,
            )
            for i in range(b):
                torch.matmul(
                    self.sampled_joint_probs_M_K,
                    chunked_log_probs_b_K_C[i].to(self.sampled_joint_probs_M_K, non_blocking=True).exp(),
                    out=probs_b_M_C[i],
                )
            probs_b_M_C /= K

            q_1_M_1 = self.sampled_joint_probs_M_K.mean(dim=1, keepdim=True)[None]

            output_entropies_B[start:end].copy_(
                torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C / q_1_M_1, dim=(1, 2)) / M,
                non_blocking=True,
            )

            pbar.update(end - start)

        pbar.close()

        return output_entropies_B


class ExactJointEntropyEstimator(MVNJointEntropyEstimator):
    def __init__(self, batch: CurrentBatch, likelihood, samples: Sampling) -> None:
        self.batch = batch
        self.likelihood = likelihood
        self.samples = samples.batch_samples
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
    def __init__(self, batch: CurrentBatch, likelihood, samples: Sampling) -> None:
        self.batch = batch
        self.likelihood = likelihood
        self.function_samples = samples.batch_samples
        self.per_samples = samples.per_samples
        self.sum_samples = samples.samples_sum
        super().__init__(batch, likelihood, samples)

    @staticmethod
    def _compute(log_probs_N_K_C: TensorType["N","K","C"], samples: int) -> TensorType:
        N, K, C = log_probs_N_K_C.shape
        print(torch.min(log_probs_N_K_C.exp()))
        batch_joint_entropy = joint_entropy.SampledJointEntropy.sample(log_probs_N_K_C.exp(), samples)

        return batch_joint_entropy.compute()

    @staticmethod
    def _compute_from_batch(batch: CurrentBatch, likelihood, samples: int, function_samples: int) -> TensorType:
        log_probs_N_K_C: TensorType["datapoints", "samples", "num_cat"] = ((likelihood(batch.distribution.sample(sample_shape=torch.Size([function_samples]))).logits)).permute(1,0,2) # type: ignore
        return BBReduxJointEntropyEstimator._compute(log_probs_N_K_C, samples)

    def compute(self) -> TensorType:
        return self._compute_from_batch(self.batch, self.likelihood, self.samples, 100)

    def compute_batch(self, pool: Rank1Updates) -> TensorType["N"]:
        N = len(pool)
        function_samples = self.function_samples
        output: TensorType["N"] = torch.zeros(N)
        pbar = tqdm(total=N, desc="BBRedux Batch", leave=False)
        samples:  TensorType["samples", "datapoints", "num_cat"] = self.batch.distribution.sample(sample_shape=torch.Size([function_samples]))

        if samples.shape[1] == 0:
            batch_joint_entropy = _SampledJointEntropy.empty(function_samples, self.likelihood)
        else:
            batch_joint_entropy = _SampledJointEntropy.sample(samples.permute(1, 0, 2), function_samples * self.per_samples, self.likelihood)
        batch_joint_entropy.compute_batch(pool, self.batch, samples, function_samples, self.per_samples, output)
        pbar.close()
        return output

    def add_variable(self, new: Rank1Update) -> None:
        self.batch = self.batch.append(new)
