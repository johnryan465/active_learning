from methods.utils import get_pool
from datasets.activelearningdataset import DatasetUtils
from models.model import UncertainModel
from models.vduq import vDUQ
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod, Method
from methods.method_params import MethodParams
from batchbald_redux.batchbald import get_batchbald_batch
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

from typeguard import check_type, typechecked
from utils.typing import MultitaskMultivariateNormalType, MultivariateNormalType, TensorType

import torch
from tqdm import tqdm
from dataclasses import dataclass
from toma import toma

from .mvn_joint_entropy import MVNJointEntropy, chunked_distribution



@dataclass
class BatchBALDParams(MethodParams):
    samples: int
    use_cuda: bool
    var_reduction: bool = True
    efficent: bool = True


# \sigma_{BatchBALD} ( {x_1, ..., x_n}, p(w)) = H(y_1, ..., y_n) - E_{p(w)} H(y | x, w)

@typechecked
def get_features(inputs: TensorType["datapoints", "channels", "x", "y"], feature_size: int, model_wrapper: vDUQ) -> TensorType["datapoints", "num_features"]:
    N = inputs.shape[0]
    pool = torch.empty((N, feature_size))
    pbar = tqdm(total=N, desc="Feature Extraction", leave=False)
    @toma.execute.chunked(inputs, N)
    def compute(inputs, start: int, end: int):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            tmp = model_wrapper.model.feature_extractor.forward(inputs).detach()
            pool[start:end].copy_(tmp, non_blocking=True)
        pbar.update(end - start)
    pbar.close()
    return pool


@typechecked
def get_gp_output(features: TensorType[ ..., "num_points", "num_features"], model_wrapper: vDUQ) -> MultivariateNormalType[("N"), ("num_points", "num_cats")]:
    # We need to expand the dimensions of the features so we can broadcast with the GP
    if len(features.shape) > 2: # we have batches
        features = features.unsqueeze(-3)
    with torch.no_grad():
        dists = []
        N = features.shape[0]
        pbar = tqdm(total=N, desc="GP", leave=False)
        @toma.execute.chunked(features, N)
        def compute(features, start: int, end: int):
            if torch.cuda.is_available():
                features = features.cuda()
            d = model_wrapper.model.gp(features)
            dists.append(d)
            pbar.update(end - start)
        pbar.close()
        return MVNJointEntropy.combine_mtmvns(dists)


@typechecked
def compute_conditional_entropy_mvn(distribution: MultitaskMultivariateNormalType[("N"), (1, "num_cats")], likelihood, num_samples : int) -> TensorType["N"]:
    # The distribution input is a batch of MVNS
    N = distribution.batch_shape[0]
    def func(dist: MultitaskMultivariateNormalType) -> TensorType:
        log_probs_K_n_C = (likelihood(dist.sample(sample_shape=torch.Size([num_samples]))).logits).squeeze()
        log_probs_n_K_C = log_probs_K_n_C.permute(1, 0, 2)
        _, K, _ = log_probs_n_K_C.shape
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)
        return (-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
    
    entropies_N = torch.empty(N, dtype=torch.double)
    chunked_distribution("Conditional Entropy", distribution, func, entropies_N)

    return entropies_N




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
                redux_candidate_indices = []
                redux_candidate_scores = []
                samples = self.params.samples
                efficent = self.params.efficent
                num_cat = 10
                feature_size = 256

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

                    features_expanded: TensorType["N", 1, "num_features"] = pool[:,None,:]
                    ind_dists: MultitaskMultivariateNormalType[("N"), (1, "num_cats")] = get_gp_output(features_expanded, model_wrapper)
                    conditional_entropies_N: TensorType["datapoints"] = compute_conditional_entropy_mvn(ind_dists, model_wrapper.likelihood, samples).cpu()
                    current_batch_dist: MultitaskMultivariateNormalType[ (), ("current_batch_size", "num_cat")] = None
                    joint_entropy_class = None

                    print(conditional_entropies_N)

                    for i in tqdm(range(batch_size), desc="Aquiring", leave=False):
                        # First we compute the joint distribution of each of the datapoints with the current aquisition
                        # We first calculate the aquisition by itself first.

                        joint_entropy_result: TensorType["datapoints"] = torch.empty(N, dtype=torch.double, pin_memory=self.params.use_cuda)
                        if i == 0:
                            # We get the current selected datapoints and broadcast them together with
                            # the pool
                            z: TensorType["current_batch_size", "num_features"] = pool[candidate_indices]
                            z: TensorType[1, "current_batch_size", "num_features"]= z[None,:,:]
                            z: TensorType["datapoints", "current_batch_size", "num_features"] = z.expand(N, -1, -1)

                            t: TensorType["datapoints", 1, "num_features"] = pool[:,None,:]
                            grouped_pool: TensorType["datapoints", "new_batch_size", "num_features"] = torch.cat([z,t], dim=1)
                            dists: MultitaskMultivariateNormalType[("chunked"), ("new_batch_size", "num_cat")] = get_gp_output(grouped_pool, model_wrapper)
                            del grouped_pool
                        
                        else:
                            # Add random sampling for larger aquisition sizes
                            # We can cache the size 2 distributions between the aquisitions (TODO)
                            # We can keep the current batch distribution from the prevoius aquisition (TODO)
                            # We perform rank-1 updates to the covariance and mean to get the new distributions 
                            # If we are performing variance reduction we could possible even make this cheaper

                            # Things to improve performance
                            # 1) caching of the feature tensors
                            # 2) don't recompute the distributions of things we have already calculated
                            # 3) Use much cleverer matrix ops on the join_rank_2 function

                            dists: MultitaskMultivariateNormalType[ ("datapoints"), ("new_batch_size", "num_cat")] = joint_entropy_class.join_rank_2() 

                        MVNJointEntropy.compute(dists, model_wrapper.likelihood, samples, joint_entropy_result, variance_reduction=self.params.var_reduction)

                        # Then we compute the batchbald objective

                        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

                        scores_N = joint_entropy_result.detach().cpu()
                        # print(scores_N)

                        scores_N -= conditional_entropies_N + shared_conditinal_entropies
                        scores_N[candidate_indices] = -float("inf")
                        # print(scores_N)
                        candidate_score, candidate_index = scores_N.max(dim=0)
                        
                        del scores_N
                        del joint_entropy_result
                        del shared_conditinal_entropies
                        del dists

                        candidate_indices.append(candidate_index.item())
                        candidate_scores.append(candidate_score.item())
                        
                        current_batch_dist: MultitaskMultivariateNormalType[ (1, 1), (2, "num_cat")] = get_gp_output( (pool[candidate_indices])[None,:,:], model_wrapper)
                        expanded_pool_features: TensorType["datapoints", 1, 1, "num_features"] = pool[:, None, None, :]
                        new_candidate_features: TensorType["datapoints", 1, 1, "num_features"] = ((pool[candidate_index])[None, None, None, :]).expand(N, -1, -1, -1)
                        joint_features: TensorType["datapoints", 1, 2, "num_features"] = torch.cat([new_candidate_features, expanded_pool_features], dim=2)
                        new_rank_2: MultitaskMultivariateNormalType[ ("datapoints", 1), (2, "num_cat")] = get_gp_output(joint_features, model_wrapper)

                        if i == 0:
                            joint_entropy_class = MVNJointEntropy(current_batch_dist, new_rank_2, samples)
                        else:
                            joint_entropy_class.add_new(current_batch_dist, new_rank_2)

                    if self.params.smoke_test:
                        efficent_candidate_indices = candidate_indices.copy()
                        efficent_candidate_scores = candidate_scores.copy()
                        # candidate_indices = []
                        # candidate_scores = []
                    
                if False:
                    # We use the BatchBALD Redux as a comparision, this does not scale to larger pool sizes.
                    pool_expanded: TensorType[1, "datapoints", "num_features"] = pool[None,:,:]
                    # joint_distribution_list: MultitaskMultivariateNormalType[(1), ("datapoints", "num_cat")] = get_gp_output(pool_expanded, model_wrapper)
                    # assert(len(joint_distribution_list) == 1)
                    joint_distribution: MultitaskMultivariateNormalType = get_gp_output(pool_expanded, model_wrapper)
                    log_probs_N_K_C: TensorType["datapoints", "samples", "num_cat"] = ((model_wrapper.likelihood(joint_distribution.sample(sample_shape=torch.Size([samples]))).logits).squeeze(1)).permute(1,0,2) # type: ignore
                    batch = get_batchbald_batch(log_probs_N_K_C, batch_size, 100000) 
                    candidate_indices = batch.indices
                    candidate_scores = batch.scores

                    if self.params.smoke_test:
                        redux_candidate_indices = candidate_indices.copy()
                        redux_candidate_scores = candidate_scores.copy()

                
                if self.params.smoke_test:
                    print("Efficent")
                    print(efficent_candidate_indices) # type: ignore
                    print(efficent_candidate_scores) # type: ignore
                    for idx in efficent_candidate_indices: # type: ignore
                        _, y = dataset.get_pool_tensor()[idx] # type: ignore
                        print(y)

                    print("Redux")
                    print(redux_candidate_indices) # type: ignore
                    print(redux_candidate_scores) # type: ignore
                    for idx in redux_candidate_indices: # type: ignore
                        _, y = dataset.get_pool_tensor()[idx] # type: ignore
                        print(y)
                Method.log_batch(dataset.get_indexes(candidate_indices), tb_logger, self.current_aquisition)
                dataset.move(candidate_indices)

                self.current_aquisition += 1
                if torch.cuda.is_available():
                    print(torch.cuda.memory_allocated())
                    print(torch.cuda.memory_reserved())
                    torch.cuda.empty_cache()

            else:
                raise NotImplementedError("BatchBALD")


    def initialise(self, dataset: ActiveLearningDataset) -> None:
        DatasetUtils.balanced_init(dataset, self.params.initial_size)

    def complete(self) -> bool:
        return self.current_aquisition >= self.params.max_num_aquisitions
