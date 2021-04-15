from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from torch import distributions
from torch.distributions.normal import Normal
from torch.tensor import Tensor
from datasets.activelearningdataset import DatasetUtils
from models.model import UncertainModel
from models.vduq import vDUQ
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod, Method
from methods.method_params import MethodParams
from batchbald_redux.batchbald import get_batchbald_batch, CandidateBatch
from batchbald_redux import joint_entropy
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

from gpytorch.lazy import CatLazyTensor, BlockDiagLazyTensor
from torch.distributions.utils import _standard_normal
from typing import Annotated, Callable, Generic, List, NewType, Tuple, Type, TypeVar, Union
from typeguard import check_type, typechecked
from utils.typing import MultitaskMultivariateNormalType, MultivariateNormalType, TensorType

import torch
from tqdm import tqdm
from dataclasses import dataclass
from toma import toma
import string



@dataclass
class BatchBALDParams(MethodParams):
    samples: int
    use_cuda: bool
    var_reduction: bool = True



# \sigma_{BatchBALD} ( {x_1, ..., x_n}, p(w)) = H(y_1, ..., y_n) - E_{p(w)} H(y | x, w)
@typechecked
def combine_mtmvns(mvns) -> MultitaskMultivariateNormalType:
    if len(mvns) < 2:
        return mvns[0]

    if not all(m.event_shape == mvns[0].event_shape for m in mvns[1:]):
        raise ValueError("All MultivariateNormals must have the same event shape")
    mean = torch.cat([mvn.mean for mvn in mvns], 0)

    covar_blocks_lazy = CatLazyTensor(
        *[mvn.lazy_covariance_matrix for mvn in mvns], dim=0, output_device=mean.device
    )
    covar_lazy = BlockDiagLazyTensor(covar_blocks_lazy, block_dim=0)
    return MultitaskMultivariateNormal(mean=mean, covariance_matrix=covar_blocks_lazy, interleaved=False)


# We compute the joint entropy of the distributions
@typechecked
def joint_entropy_mvn(distributions: List[MultivariateNormalType], likelihood, per_samples: int, output: TensorType["N"], variance_reduction: bool = False) -> None:
    # We can exactly compute a larger sized exact distribution
    # As the task batches are independent we can chunk them
    D = distributions[0].event_shape[0]
    N = sum([distribution.batch_shape[0] for distribution in distributions])
    t = string.ascii_lowercase[:D]
    s =  ','.join(['yz' + c for c in list(t)]) + '->' + 'yz' + t

    if D < 5:
        # We can chunk to get away with lower memory usage
        # We can't lazily split, only lazily combine
        if variance_reduction:
            # here is where we generate the samples which we will 
            shape = [per_samples] + list(distributions[0].base_sample_shape)
            samples = _standard_normal(torch.Size(shape), dtype=distributions[0].loc.dtype, device=distributions[0].loc.device)

        @typechecked
        def func(distribution: MultivariateNormalType) -> TensorType["N"]:
            if variance_reduction:
                base_samples = samples.detach().clone()
                base_samples = base_samples[:,None,:,:]
                base_samples = base_samples.expand(-1, distribution.batch_shape[0], -1, -1)
                l: TensorType["num_samples", "N", "num_points", "num_cat"] = likelihood(distribution.sample(base_samples=base_samples)).probs
            else:
                l: TensorType["num_samples", "N", "num_points", "num_cat"] = likelihood(distribution.sample(sample_shape=torch.Size([per_samples]))).probs

            l: TensorType["N", "num_samples", "num_points", "num_cat"] = torch.transpose(l, 0, 1)
            j: List[TensorType["N", "num_samples", "num_cat"]] = list(torch.unbind(l, dim=-2))
            g: TensorType["N", "num_samples", "expanded" : ...] = torch.einsum(s, *j) # We should have num_points dimensions each of size num_cat
            g: TensorType["N", "expanded" : ...] = torch.mean(g, 1)
            g: TensorType["N", "expanded" : ...] = g * torch.log(g)
            g: TensorType["N"] = - torch.sum(g.flatten(1), 1)
            return g
        
        chunked_distribution("Joint Entropy", distributions, func, output)
    else:
        return None

@typechecked
def chunked_distribution(name: str, distributions: List[MultivariateNormalType], func: Callable, output: TensorType["N": ...]) -> None:
    N = output.shape[0]
    len_d = len(distributions)
    @toma.batch(initial_batchsize=len_d)
    def compute(batchsize: int, distributions: List[MultivariateNormalType]):
        pbar = tqdm(total=N, desc=name, leave=False)
        start = 0
        end = 0
        for i in range(0, len_d, batchsize):
            distribution = combine_mtmvns(distributions[i: min(i+batchsize, len_d)])
            end = start + distribution.batch_shape[0]
            g = func(distribution)
            output[start:end].copy_(g)
            pbar.update(end - start)
            start = end
            pbar.close()
    compute(distributions) #type: ignore

# Gets the pool from the dataset as a tensor
@typechecked
def get_pool(dataset: ActiveLearningDataset) -> TensorType["datapoints", "channels", "x", "y"]:
    inputs = []
    for x, _ in tqdm(dataset.get_pool_tensor(), desc="Loading pool", leave=False):
        inputs.append(x)
    inputs = torch.stack(inputs, dim=0)
    return inputs


@typechecked
def get_features(inputs: TensorType["datapoints", "channels", "x", "y"], feature_size: int, model_wrapper: vDUQ) -> TensorType["datapoints", "num_features"]:
    N = inputs.shape[0]
    pool = torch.empty((N, feature_size))
    pbar = tqdm(total=N, desc="Feature Extraction", leave=False)
    @toma.execute.chunked(inputs, 1024)
    def compute(inputs, start: int, end: int):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            tmp = model_wrapper.model.feature_extractor.forward(inputs).detach()
            pool[start:end].copy_(tmp)
        pbar.update(end - start)
    pbar.close()
    return pool


@typechecked
def get_gp_output(features: TensorType[ ..., "num_points", "num_features"], model_wrapper: vDUQ) -> List[MultivariateNormalType[("chunk_size"), ("num_points", "num_cats")]]:
    # We need to expand the dimensions of the features so we can broadcast with the GP
    if len(features.shape) > 2: # we have batches
        features = features.unsqueeze(-3)
    with torch.no_grad():
        dists = []
        N = features.shape[0]
        pbar = tqdm(total=N, desc="GP", leave=False)
        @toma.execute.chunked(features, 256)
        def compute(features, start: int, end: int):
            if torch.cuda.is_available():
                features = features.cuda()
            d = model_wrapper.model.gp(features)
            dists.append(d)
            pbar.update(end - start)
        pbar.close()
        return dists


@typechecked
def compute_conditional_entropy_mvn(distributions: List[MultitaskMultivariateNormalType[("chunk_size"), (1, "num_cats")]], likelihood, num_samples : int) -> TensorType["datapoints"]:
    # The distribution input is a batch of MVNS
    N = sum(map(lambda x: x.batch_shape[0], distributions))
    log_probs_N_K_C = torch.empty(N, num_samples, 10)
    def func(distribution: MultitaskMultivariateNormalType) -> TensorType:
        h = (likelihood(distribution.sample(sample_shape=torch.Size([num_samples]))).logits).squeeze()
        h = h.permute(1, 0, 2)
        return h
    chunked_distribution("Sampling", distributions, func, log_probs_N_K_C)
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


@typechecked
def join_rank_2(candidate_dist: MultitaskMultivariateNormalType[(),("batch_size", "num_cat")], rank_2_dists: MultitaskMultivariateNormalType[("datapoints","batch_size"), (2, "num_cat")]) -> List[MultitaskMultivariateNormalType[("chunked"),("new_batch_size","num_cat")]]:
    # For each of the datapoints and the candidate batch we want to compute the low rank tensor
    # The order of the candidate datapoints must be maintained and used carefully
    # Need to wrap this in toma
    batch_size = candidate_dist.event_shape[0]
    num_cats = candidate_dist.event_shape[1]
    num_datapoints = rank_2_dists.batch_shape[0]
    distributions = []
    cov = rank_2_dists.covariance_matrix

    item_means = rank_2_dists.mean[:,1,1,:].unsqueeze(1)
    candidate_means = candidate_dist.mean[None,:,:].expand(num_datapoints, -1, -1)
    mean_tensor = torch.cat([candidate_means, item_means], dim=1)

    expanded_batch_covar = candidate_dist.lazy_covariance_matrix.expand(num_datapoints, batch_size*num_cats, batch_size*num_cats)
    self_covar_tensor = cov[:, 0, num_cats:, num_cats:]
    cross_mat = torch.cat(torch.unbind(cov[:,:, :num_cats, num_cats:], dim=1), dim=-1)
    covar_tensor = expanded_batch_covar.cat_rows(cross_mat, self_covar_tensor)
    
    group_size = 256
    for start in tqdm(range(0, num_datapoints, group_size), desc="Joining", leave=False):
        end = min((start + group_size), num_datapoints)
        new_dist = MultitaskMultivariateNormal(mean_tensor[start:end], covar_tensor[start:end])
        distributions.append(new_dist)
    return distributions


class BatchBALD(UncertainMethod):
    def __init__(self, params: BatchBALDParams) -> None:
        super().__init__()
        self.params = params
        self.current_aquisition = 0

    @typechecked
    def acquire(self, model_wrapper: UncertainModel, dataset: ActiveLearningDataset, tb_logger: TensorboardLogger) -> None:
        with torch.no_grad():
            if isinstance(model_wrapper, vDUQ):
                candidate_indices = []
                candidate_scores = []
                samples = self.params.samples
                efficent = True
                num_cat = 10
                feature_size = 512

                inputs: TensorType["datapoints","channels","x","y"] = get_pool(dataset)
                N = inputs.shape[0]

                pool: TensorType["datapoints","num_features"] = get_features(inputs, feature_size, model_wrapper)

                model_wrapper.model.eval()
                batch_size = self.params.aquisition_size
                batch_size = min(batch_size, N)

                if batch_size == 0:
                    self.current_aquisition += 1
                    return
                
                if efficent or self.params.smoke_test:
                    # We cant use the standard get_batchbald_batch function as we would need to sample and entire function from posterior
                    # which is computationaly prohibative (has complexity related to the pool size)

                    # We instead need to repeatedly compute the updated probabilties for each aquisition
                    
                    # We can instead of recomputing the entire distribtuion, we can compute all the pairs with the elements of the candidate batch
                    # We can use this to build the new distributions for batch size
                    # We will not directly manipulate the inducing points as there are various different strategies.
                    # Instead we will we take advantage of the fact that GP output is a MVN and can be conditioned.

                    features_expanded: TensorType["datapoints", 1, "num_features"] = pool[:,None,:]
                    ind_dists: List[MultitaskMultivariateNormalType[("chunk_size"), (1, "num_cats")]] = get_gp_output(features_expanded, model_wrapper)
                    conditional_entropies_N: TensorType["datapoints"] = compute_conditional_entropy_mvn(ind_dists, model_wrapper.likelihood, samples).cpu()
                    current_batch_dist: MultitaskMultivariateNormalType[ (), ("current_batch_size", "num_cat")] = None

                    for i in tqdm(range(batch_size), desc="Aquiring", leave=False):
                        # First we compute the joint distribution of each of the datapoints with the current aquisition
                        # We first calculate the aquisition by itself first.

                        joint_entropy_result: TensorType["datapoints"] = torch.empty(N, dtype=torch.double, pin_memory=self.params.use_cuda)

                        if True:
                            # We get the current selected datapoints and broadcast them together with
                            # the pool
                            z: TensorType["current_batch_size", "num_features"] = pool[candidate_indices]
                            z: TensorType[1, "current_batch_size", "num_features"]= z[None,:,:]
                            z: TensorType["datapoints", "current_batch_size", "num_features"] = z.expand(N, -1, -1)

                            t: TensorType["datapoints", 1, "num_features"] = pool[:,None,:]
                            grouped_pool: TensorType["datapoints", "new_batch_size", "num_features"] = torch.cat([z,t], dim=1)

                            dists: List[MultitaskMultivariateNormalType[("chunked"), ("new_batch_size", "num_cat")]] = get_gp_output(grouped_pool, model_wrapper)
                        
                        else:
                            candidate_points: TensorType["current_batch_size", "num_features"] = pool[candidate_indices]

                            # Add random sampling for larger aquisition sizes
                            # We can cache the size 2 distributions between the aquisitions (TODO)
                            # We can keep the current batch distribution from the prevoius aquisition (TODO)
                            # We perform rank-1 updates to the covariance and mean to get the new distributions 
                            # If we are performing variance reduction we could possible even make this cheaper

                            # Things to improve performance
                            # 1) caching of the feature tensors
                            # 2) don't recompute the distributions of things we have already calculated
                            # 3) Use much cleverer matrix ops on the join_rank_2 function

                            expanded_candidate_features: TensorType[1, "current_batch_size", 1, "num_features"] = (pool[candidate_indices])[None,:,None,:]
                            repeated_candidate_features: TensorType["datapoints", "current_batch_size", 1, "num_features"] = expanded_candidate_features.expand(N, -1, -1, -1)
                            expanded_pool_features: TensorType["datapoints", 1, 1, "num_features"] = pool[:, None, None, :]
                            repeated_pool_features: TensorType["datapoints", "current_batch_size", 1, "num_features"] = expanded_pool_features.expand(-1, i, -1, -1)
                            combined_features: TensorType["datapoints", "current_batch_size", 2, "num_features"] = torch.cat([repeated_candidate_features, repeated_pool_features], dim=2)
                            chunked_dist: List[MultitaskMultivariateNormalType[ ("datapoints", "current_batch_size"), (2, "num_cat")]] = get_gp_output(combined_features, model_wrapper)
                            size_2_dists: MultitaskMultivariateNormalType[ ("datapoints", "current_batch_size"), (2, "num_cat")] = combine_mtmvns(chunked_dist)
                            dists: List[MultitaskMultivariateNormalType[ ("chunked"), ("new_batch_size", "num_cat")]] = join_rank_2(current_batch_dist, size_2_dists) 

                        joint_entropy_mvn(dists, model_wrapper.likelihood, samples, joint_entropy_result, variance_reduction=self.params.var_reduction)
                        if self.params.smoke_test:
                            print(joint_entropy_result)

                        # Then we compute the batchbald objective

                        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()
                        print(shared_conditinal_entropies)

                        scores_N = joint_entropy_result.detach().cpu()

                        scores_N -= conditional_entropies_N + shared_conditinal_entropies
                        scores_N[candidate_indices] = -float("inf")

                        candidate_score, candidate_index = scores_N.max(dim=0)
                        
                        candidate_indices.append(candidate_index.item())
                        candidate_scores.append(candidate_score.item())
                        
                        current_batch_dist = combine_mtmvns(get_gp_output(pool[candidate_indices], model_wrapper)) 
                    if self.params.smoke_test:
                        efficent_candidate_indices = candidate_indices.copy()
                        efficent_candidate_scores = candidate_scores.copy()
                        candidate_indices = []
                        candidate_scores = []
                    
                if not efficent or self.params.smoke_test:
                    # We use the BatchBALD Redux as a comparision, this does not scale to larger pool sizes.
                    pool_expanded: TensorType[1, "datapoints", "num_features"] = pool[None,:,:]
                    joint_distribution_list: List[MultitaskMultivariateNormalType[(1), ("datapoints", "num_cat")]] = get_gp_output(pool_expanded, model_wrapper)
                    assert(len(joint_distribution_list) == 1)
                    joint_distribution: MultitaskMultivariateNormalType = joint_distribution_list[0]
                    log_probs_N_K_C: TensorType["datapoints", "samples", "num_cat"] = ((model_wrapper.likelihood(joint_distribution.sample(sample_shape=torch.Size([samples]))).logits).squeeze(1)).permute(1,0,2)
                    batch = get_batchbald_batch(log_probs_N_K_C, batch_size, samples) 
                    candidate_indices = batch.indices
                    candidate_scores = batch.scores

                    if self.params.smoke_test:
                        redux_candidate_indices = candidate_indices.copy()
                        redux_candidate_scores = candidate_scores.copy()

                
                if self.params.smoke_test:
                    print("Efficent")
                    print(efficent_candidate_indices)
                    print(efficent_candidate_scores)

                    print("Redux")
                    print(redux_candidate_indices)
                    print(redux_candidate_scores)
                Method.log_batch(dataset.get_indexes(candidate_indices), tb_logger, self.current_aquisition)
                dataset.move(candidate_indices)

                self.current_aquisition += 1

            else:
                raise NotImplementedError("BatchBALD")

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        DatasetUtils.balanced_init(dataset, self.params.initial_size)

    def complete(self) -> bool:
        return self.current_aquisition >= self.params.max_num_aquisitions
