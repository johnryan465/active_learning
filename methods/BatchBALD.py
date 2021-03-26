from datasets.activelearningdataset import DatasetUtils
from models.model import UncertainModel
from models.vduq import vDUQ
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod
from methods.method_params import MethodParams
from batchbald_redux.batchbald import get_batchbald_batch, CandidateBatch
from batchbald_redux import joint_entropy

from gpytorch.distributions import MultivariateNormal
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


# \sigma_{BatchBALD} ( {x_1, ..., x_n}, p(w)) = H(y_1, ..., y_n) - E_{p(w)} H(y | x, w)


def joint_entropy_mvn(distribution : MultivariateNormal, likelihood, per_samples, num_configs) -> torch.tensor:
    # We need to compute
    if distribution.event_shape[0] < 5:
        l = likelihood(distribution.sample(sample_shape=torch.Size([per_samples]))).probs
        l = torch.transpose(l, 0, 1)
        t = string.ascii_lowercase[:distribution.event_shape[0]]
        s =  ','.join(['z' + c for c in list(t)]) + '->' + 'z' + t
        g = torch.einsum(s, *torch.unbind(l))
        g = torch.mean(g, dim=0)
        return -torch.sum(g * torch.log(g))
    
    else:
        return torch.tensor(0)

def compute_conditional_entropy_mvn(distributions: List[MultivariateNormal], likelihood, num_samples : int) -> torch.Tensor:
    log_probs_N_K_C = torch.stack([likelihood(distribution.sample(sample_shape=torch.Size([num_samples]))).logits for distribution in distributions], dim=0).squeeze()
    # print(log_probs_N_K_C.shape)
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

    def acquire(self, model: UncertainModel, dataset: ActiveLearningDataset) -> None:
        if isinstance(model, vDUQ):
            # We cant use the standard get_batchbald_batch function as we would need to sample and entire function from posterior
            # which is computationaly prohibative (has complexity related to the pool size)

            # We instead need to repeatedly compute the updated probabilties for each aquisition
            pool = []
            samples = 500
            num_configs = 10
            count = 0
            for x, i in tqdm(dataset.get_pool(), desc="Loading pool", leave=False):
                if self.params.use_cuda:
                    x = x.cuda()
                pool.append(model.feature_extractor.forward(x).detach().clone())
                count = count + 1
                if count > 5:
                   break

            pool = torch.cat(pool, dim=0)
            # print(pool.shape)
            N = pool.shape[0]
            batch_size = self.params.aquisition_size
            batch_size = min(batch_size, N)

            if batch_size == 0:
                self.current_aquisition += 1
                return

            candidate_indices = []
            candidate_scores = []

            conditional_entropies_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())
            dists = []
            for j in range(0, N):
                x = (pool[j])[None, :,]
                #mu_x = model.model.gp.mean_module(x)
                # k_xz = model.model.gp.covar_module(x, z).evaluate()
                # k_xx = model.model.gp.covar_module(x, x).evaluate()
                # a = torch.cat([k_zz, k_xz], dim=1)
                # b = torch.cat([torch.transpose(k_xz,1,2),k_xx], dim=1)
                # k = torch.cat([a,b], dim=2)
                k = model.model.gp.covar_module(x, x)
                mu = model.model.gp.mean_module(x)
                dists.append(MultivariateNormal(mu, k))


            conditional_entropies_N = compute_conditional_entropy_mvn(dists, model.likelihood, samples)
            
            for i in tqdm(range(batch_size), desc="vDUQ BatchBALD", leave=False):
                # First we compute the joint distribution of each of the datapoints with the current aquisition
                # We first calculate the aquisition by itself first.
                joint_entropy_result = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())
                scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())
                # dists = torch.empty(N, dtype=torch.distributions, pin_memory=torch.cuda.is_available())
                dists = []

                if i > 0:
                    z = pool[candidate_indices]
                    #mu_z = model.model.gp.mean_module(z)
                    #k_zz = model.model.gp.covar_module(z, z).evaluate()
                    for j in range(0, N):
                        if(j % 50 == 0):
                            print(j)
                        if j in candidate_indices:
                            pass
                        else:
                            x = (pool[j])[None, :,]
                            #mu_x = model.model.gp.mean_module(x)
                            r = torch.cat( [z, x], dim=0)
                            # k_xz = model.model.gp.covar_module(x, z).evaluate()
                            # k_xx = model.model.gp.covar_module(x, x).evaluate()
                            # a = torch.cat([k_zz, k_xz], dim=1)
                            # b = torch.cat([torch.transpose(k_xz,1,2),k_xx], dim=1)
                            # k = torch.cat([a,b], dim=2)
                            k = model.model.gp.covar_module(r, r)
                            mu = model.model.gp.mean_module(r)
                            dist = MultivariateNormal(mu, k)
                            joint_entropy_result[j] = joint_entropy_mvn(dist, model.likelihood, samples, num_configs)
                            

                    # We have the descriptions of the joint gaussians which represents all the possible candidate batches
                    # of the next size
                    # candidate_indices.append(i)
                else:
                    # @toma.execute.chunked(pool, 1024)
                    # def compute(pool, start: int, end: int):
                    #     x = pool[start:end][None,:,:]
                    #     mu_x = model.model.gp.mean_module(x)
                    #     k_xx = model.model.gp.covar_module(x, x)
                    #     print(x)
                    #     # entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
                    #     joint_entropy_result[start:end].copy_(joint_entropy_mvn(dists, model.likelihood, samples))
                    #     pbar.update(end - start)
                    for j in range(0, N):
                        x = (pool[j])[None, :,]
                        mu_x = model.model.gp.mean_module(x)
                        k_xx = model.model.gp.covar_module(x, x)
                        dist = MultivariateNormal(mu_x, k_xx)
                        if(j % 50 == 0):
                            print(j)
                        joint_entropy_result[j] = joint_entropy_mvn(dist, model.likelihood, samples, num_configs)

                # Then we compute the batchbald objective

                shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

                scores_N = joint_entropy_result
                print(N)
                print(candidate_indices)
                print(scores_N.shape, conditional_entropies_N.shape, shared_conditinal_entropies)
                scores_N -= conditional_entropies_N + shared_conditinal_entropies
                scores_N[candidate_indices] = -float("inf")

                candidate_score, candidate_index = scores_N.max(dim=0)
                
                candidate_indices.append(candidate_index.item())
                candidate_scores.append(candidate_score.item())
                        
            dataset.move(candidate_indices)

        else:
            probs = []
            pool = []
            for x, _ in tqdm(dataset.get_pool(), desc="Loading pool", leave=False):
                if self.params.use_cuda:
                    x = x.cuda()
                # probs_ = model.sample(x, self.params.samples).detach().clone()
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
