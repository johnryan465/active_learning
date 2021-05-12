from uncertainty.estimator_entropy import MVNJointEntropyEstimator, Sampling
from uncertainty.current_batch import CurrentBatch
from uncertainty.rank2 import Rank1Update, Rank1Updates


from toma import toma
from tqdm import tqdm
from batchbald_redux import joint_entropy
from utils.typing import TensorType
import torch

# Here we have a class for computing the joint entropy based on snippest from the BB Redux code to ensure that the code is correct

class MVNSampledJointEntropy:
    sampled_joint_probs_M_K: torch.Tensor

    def __init__(self, sampled_joint_probs_M_K: torch.Tensor, likelihood):
        self.sampled_joint_probs_M_K = sampled_joint_probs_M_K
        self.likelihood = likelihood

    @staticmethod
    def empty(K: int, likelihood, device=None, dtype=None) -> "MVNSampledJointEntropy":
        return MVNSampledJointEntropy(torch.ones((1, K), device=device, dtype=dtype), likelihood)

    @staticmethod
    def sample(samples: torch.Tensor, M: int, likelihood) -> "MVNSampledJointEntropy":
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
        return MVNSampledJointEntropy(samples_M_K, likelihood)

    def compute(self) -> torch.Tensor:
        sampled_joint_probs_M = torch.mean(self.sampled_joint_probs_M_K, dim=1, keepdim=False)
        nats_M = -torch.log(sampled_joint_probs_M)
        entropy = torch.mean(nats_M)
        return entropy


    @staticmethod
    def _compute_batch(sampled_joint_probs_M_K: TensorType["M","K"], likelihood,  pool: Rank1Updates, batch, samples: TensorType["K", "D", "C"], per_samples: int, output_entropies_B=None):
        function_samples = samples.shape[0]
        candidate_samples = []
        for i, candidate in enumerate(pool):
            cond_dists = batch.create_conditionals_from_rank1s(Rank1Updates(already_computed=[candidate]), samples, function_samples)
            dist = next(next(cond_dists))
            sample_ = dist.sample(sample_shape=torch.Size([per_samples])).squeeze(3).squeeze(1)
            sample_ = torch.mean(sample_, dim=0, keepdim=True)
            candidate_samples.append(sample_)

        candidate_samples = torch.cat(candidate_samples, dim=0)
        
        log_probs_B_K_C = likelihood(candidate_samples).logits
        assert sampled_joint_probs_M_K.shape[1] == log_probs_B_K_C.shape[1]

        B, K, C = log_probs_B_K_C.shape
        M = sampled_joint_probs_M_K.shape[0]

        if output_entropies_B is None:
            output_entropies_B = torch.empty(B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device)

        pbar = tqdm(total=B, desc="SampledJointEntropy.compute_batch", leave=False)

        @toma.execute.chunked(log_probs_B_K_C, initial_step=1024, dimension=0)
        def chunked_joint_entropy(chunked_log_probs_b_K_C: torch.Tensor, start: int, end: int):
            b = chunked_log_probs_b_K_C.shape[0]

            probs_b_M_C = torch.empty(
                (b, M, C),
                dtype=sampled_joint_probs_M_K.dtype,
                device=sampled_joint_probs_M_K.device,
            )
            for i in range(b):
                torch.matmul(
                    sampled_joint_probs_M_K,
                    chunked_log_probs_b_K_C[i].to(sampled_joint_probs_M_K, non_blocking=True).exp(),
                    out=probs_b_M_C[i],
                )
            probs_b_M_C /= K

            q_1_M_1 = sampled_joint_probs_M_K.mean(dim=1, keepdim=True)[None]

            output_entropies_B[start:end].copy_(
                torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C / q_1_M_1, dim=(1, 2)) / M,
                non_blocking=True,
            )

            pbar.update(end - start)

        pbar.close()

        return output_entropies_B

    def compute_batch(self, pool: Rank1Updates, batch: CurrentBatch, samples: TensorType["K", "B", "C"], per_samples: int, output_entropies_B=None):
        return MVNSampledJointEntropy._compute_batch(self.sampled_joint_probs_M_K, self.likelihood, pool, batch, samples, per_samples, output_entropies_B)

class BBReduxJointEntropyEstimator(MVNJointEntropyEstimator):
    def __init__(self, batch: CurrentBatch, likelihood, samples: Sampling) -> None:
        self.batch = batch
        self.likelihood = likelihood
        self.function_samples = samples.batch_samples
        self.per_samples = samples.per_samples
        self.sum_samples = samples.sum_samples
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
        return self._compute_from_batch(self.batch, self.likelihood, self.function_samples, 100)

    def compute_batch(self, pool: Rank1Updates) -> TensorType["N"]:
        N = len(pool)
        function_samples = self.function_samples
        output: TensorType["N"] = torch.zeros(N)
        pbar = tqdm(total=N, desc="BBRedux Batch", leave=False)
        samples:  TensorType["samples", "datapoints", "num_cat"] = self.batch.distribution.sample(sample_shape=torch.Size([function_samples]))

        if samples.shape[1] == 0:
            batch_joint_entropy = MVNSampledJointEntropy.empty(function_samples, self.likelihood)
        else:
            batch_joint_entropy = MVNSampledJointEntropy.sample(samples.permute(1, 0, 2), function_samples * self.sum_samples, self.likelihood)
        batch_joint_entropy.compute_batch(pool, self.batch, samples, self.per_samples, output)
        pbar.close()
        return output

    def add_variable(self, new: Rank1Update) -> None:
        self.batch = self.batch.append(new)
