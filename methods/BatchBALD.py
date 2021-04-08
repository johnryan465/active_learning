from datasets.activelearningdataset import DatasetUtils
from models.model import UncertainModel
from models.vduq import vDUQ
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod
from methods.method_params import MethodParams
from batchbald_redux.batchbald import get_batchbald_batch, CandidateBatch
from batchbald_redux import joint_entropy
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.lazy import CatLazyTensor, BlockDiagLazyTensor
from typing import List

import torch
from tqdm import tqdm
from dataclasses import dataclass
from toma import toma
import string

@dataclass
class BatchBALDParams(MethodParams):
    samples: int
    use_cuda: bool

def combine_mtmvns(mvns):
    if len(mvns) < 2:
        raise ValueError("Must provide at least 2 MVNs to form a MultitaskMultivariateNormal")

    if not all(m.event_shape == mvns[0].event_shape for m in mvns[1:]):
        raise ValueError("All MultivariateNormals must have the same event shape")
    mean = torch.cat([mvn.mean for mvn in mvns], 0)

    covar_blocks_lazy = CatLazyTensor(
        *[mvn.lazy_covariance_matrix for mvn in mvns], dim=0, output_device=mean.device
    )
    covar_lazy = BlockDiagLazyTensor(covar_blocks_lazy, block_dim=0)
    return MultitaskMultivariateNormal(mean=mean, covariance_matrix=covar_blocks_lazy, interleaved=False)


def combine_mvns(mvns):
    if len(mvns) < 2:
        raise ValueError("Must provide at least 2 MVNs to form a MultitaskMultivariateNormal")
    if any(isinstance(mvn, MultitaskMultivariateNormal) for mvn in mvns):
        raise ValueError("Cannot accept MultitaskMultivariateNormals")
    if not all(m.event_shape == mvns[0].event_shape for m in mvns[1:]):
        raise ValueError("All MultivariateNormals must have the same event shape")
    mean = torch.cat([mvn.mean for mvn in mvns], 0)

    covar_blocks_lazy = CatLazyTensor(
        *[mvn.lazy_covariance_matrix for mvn in mvns], dim=0, output_device=mean.device
    )
    covar_lazy = BlockDiagLazyTensor(covar_blocks_lazy, block_dim=0)
    return MultitaskMultivariateNormal(mean=mean, covariance_matrix=covar_lazy, interleaved=False)

# \sigma_{BatchBALD} ( {x_1, ..., x_n}, p(w)) = H(y_1, ..., y_n) - E_{p(w)} H(y | x, w)


# We compute the 
def joint_entropy_mvn(distribution : MultivariateNormal, likelihood, per_samples, num_configs : int) -> torch.Tensor:
    # We wish to find the indexs which we are sampling from
    D = distribution.event_shape[1]
    if D < 5:
        l = likelihood(distribution.sample(sample_shape=torch.Size([per_samples]))).probs
        l = torch.transpose(l, 0, 1)
        t = string.ascii_lowercase[:D]
        s =  ','.join(['yz' + c for c in list(t)]) + '->' + 'yz' + t
        g = torch.einsum(s, *torch.unbind(l, dim=2))
        g = g * torch.log(g)
        g = -torch.sum(g, dim=tuple(range(2,2+D)))
        g = torch.mean(g, dim=1)
        return g
    else:
        return torch.zeros(distribution.batch_shape)

def compute_conditional_entropy_mvn(distributions: MultivariateNormal, likelihood, num_samples : int) -> torch.Tensor:
    # The distribution input is a batch of MVNS
    log_probs_N_K_C = (likelihood(distributions.sample(sample_shape=torch.Size([num_samples]))).logits).permute(1, 0, 2)
    N, K, C = log_probs_N_K_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)
    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N

class BatchBALD(UncertainMethod):
    def __init__(self, params: BatchBALDParams) -> None:
        super().__init__()
        self.params = params
        self.current_aquisition = 0

    def acquire(self, model: UncertainModel, dataset: ActiveLearningDataset, tb_logger: TensorboardLogger) -> None:
        if isinstance(model, vDUQ):
            # We cant use the standard get_batchbald_batch function as we would need to sample and entire function from posterior
            # which is computationaly prohibative (has complexity related to the pool size)

            # We instead need to repeatedly compute the updated probabilties for each aquisition
            
            samples = 100
            num_cat = 10
            feature_size = 512

            inputs = []

            for x, i in tqdm(dataset.get_pool(), desc="Loading pool", leave=False):
                inputs.append(x)
            inputs = torch.cat(inputs, dim=0)

            N = inputs.shape[0]
            pool = torch.empty((N, feature_size))
            pbar = tqdm(total=N, desc="Feature Extraction", leave=False)

            @toma.execute.chunked(inputs, 1024)
            def compute(inputs, start: int, end: int):
                with torch.no_grad():
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                    tmp = model.feature_extractor.forward(inputs).detach()
                    pool[start:end].copy_(tmp)
                pbar.update(end - start)

            pbar.close()

            model.model.eval()
            batch_size = self.params.aquisition_size
            batch_size = min(batch_size, N)

            if batch_size == 0:
                self.current_aquisition += 1
                return

            candidate_indices = []
            candidate_scores = []

            conditional_entropies_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

            dists = []

            pbar = tqdm(total=N, desc="Conditional Distributions", leave=False)
            @toma.execute.chunked(pool, 256)
            def compute(pool, start: int, end: int):
                with torch.no_grad():
                    if torch.cuda.is_available():
                        pool = pool.cuda()
                    d = model.model.gp(pool).to_data_independent_dist()
                    dists.append(d)
                pbar.update(end - start)
            pbar.close()
            
            dists = combine_mvns(dists).to_data_independent_dist()
            conditional_entropies_N = compute_conditional_entropy_mvn(dists, model.likelihood, samples).cpu()
            
            for i in tqdm(range(batch_size), desc="Aquiring", leave=False):
                # First we compute the joint distribution of each of the datapoints with the current aquisition
                # We first calculate the aquisition by itself first.
                joint_entropy_result = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())
                scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

                if i > 0:
                    z = pool[candidate_indices]
                    z = z[None,:,:]
                    z = z.expand(N, -1, -1)

                    dists = []
                    pbar = tqdm(total=N, desc="Joint", leave=False)
                    t = pool[:,None,:]
                    grouped_pool = torch.cat([z,t], dim=1)
                    grouped_pool = grouped_pool[:,None,:,:]

                    @toma.execute.chunked(grouped_pool, 256)
                    def compute(grouped_pool, start: int, end: int):
                        with torch.no_grad():
                            if torch.cuda.is_available():
                                grouped_pool = grouped_pool.cuda()
                            d = model.model.gp(grouped_pool)
                            dists.append(d)
                        pbar.update(end - start)
                    pbar.close()

                    dists_ = combine_mtmvns(dists)
                    joint_entropy_result = joint_entropy_mvn(dists_, model.likelihood, samples, num_cat)
                            
                else:
                    dists = []
                    pbar = tqdm(total=N, desc="Joint", leave=False)
                    @toma.execute.chunked(pool, 256)
                    def compute(pool, start: int, end: int):
                        with torch.no_grad():
                            if torch.cuda.is_available():
                                pool = pool.cuda()
                            d = model.model.gp(pool).to_data_independent_dist()
                            dists.append(d)
                        pbar.update(end - start)
                    pbar.close()
                    dists = combine_mvns(dists)
                    dists_ind = dists.to_data_independent_dist()
                    dists_ind_ = MultitaskMultivariateNormal.from_repeated_mvn(dists_ind, 1)
                    joint_entropy_result = joint_entropy_mvn(dists_ind_, model.likelihood, samples, num_cat)

                # Then we compute the batchbald objective

                shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

                scores_N = joint_entropy_result.cpu()

                scores_N -= conditional_entropies_N + shared_conditinal_entropies
                scores_N[candidate_indices] = -float("inf")

                candidate_score, candidate_index = scores_N.max(dim=0)
                
                candidate_indices.append(candidate_index.item())
                candidate_scores.append(candidate_score.item())
            
            Method.log_batch(dataset.get_indexes(candidate_indices), tb_logger, self.current_aquisition)
            dataset.move(candidate_indices)
            self.current_aquisition += 1

        else:
            probs = []
            pool = []
            for x, _ in tqdm(dataset.get_pool(), desc="Loading pool", leave=False):
                if self.params.use_cuda:
                    x = x.cuda()
                pool.append(model.feature_extractor.forward(x).detach().clone())

            pool = torch.cat(pool, dim=0)
            mean = model.model.gp.mean_module(pool)
            covar = model.model.gp.covar_module(pool)
            dist = MultivariateNormal(mean, covar)

            probs = model.sample_gp(pool, self.params.samples)
            batch = get_batchbald_batch(probs, self.params.aquisition_size, self.params.samples)
            dataset.move(batch.indices)
            self.current_aquisition += 1

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        DatasetUtils.balanced_init(dataset, self.params.initial_size)

    def complete(self) -> bool:
        return self.current_aquisition >= self.params.max_num_aquisitions
