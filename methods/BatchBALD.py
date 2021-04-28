from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from gpytorch.lazy.cat_lazy_tensor import CatLazyTensor
from gpytorch.likelihoods import likelihood
from methods.utils import get_pool
from datasets.activelearningdataset import DatasetUtils
from models.model import UncertainModel
from models.vduq import vDUQ
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod, Method
from methods.method_params import MethodParams
from batchbald_redux.batchbald import compute_conditional_entropy, get_batchbald_batch
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

from typeguard import check_type, typechecked
from utils.typing import MultitaskMultivariateNormalType, MultivariateNormalType, TensorType

import torch
from tqdm import tqdm
from dataclasses import dataclass
from toma import toma

from .mvn_joint_entropy import GPCJointEntropy, LowMemMVNJointEntropy, MVNJointEntropy, Rank2Next, chunked_distribution



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
        # We want to keep things off the GPU
        dist = GPCJointEntropy.combine_mtmvns(dists)
        mean_cpu = dist.mean
        cov_cpu = dist.lazy_covariance_matrix
        if torch.cuda.is_available():
            if(isinstance(mean_cpu, CatLazyTensor)):
                mean_cpu = mean_cpu.all_to("cpu")
            else:
                mean_cpu = mean_cpu.cpu()

            if(isinstance(cov_cpu, CatLazyTensor)):
                cov_cpu = cov_cpu.all_to("cuda")
            else:
                cov_cpu = cov_cpu.cuda()
        # mean_cpu = dist.mean.cpu()
        # cov_cpu = dist.lazy_covariance_matrix.cpu()
        return MultitaskMultivariateNormal(mean=mean_cpu, covariance_matrix=cov_cpu)


@typechecked
def compute_conditional_entropy_mvn(distribution: MultitaskMultivariateNormalType[("N"), (1, "num_cats")], likelihood, num_samples : int) -> TensorType["N"]:
    # The distribution input is a batch of MVNS
    N = distribution.batch_shape[0]
    def func(dist: MultitaskMultivariateNormalType) -> TensorType:
        log_probs_K_n_C = (likelihood(dist.sample(sample_shape=torch.Size([num_samples]))).logits).squeeze()
        log_probs_n_K_C = log_probs_K_n_C.permute(1, 0, 2)
        return compute_conditional_entropy(log_probs_N_K_C=log_probs_n_K_C)
    
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
                use_bb_redux = self.params.smoke_test

                inputs: TensorType["datapoints","channels","x","y"] = get_pool(dataset)
                N = inputs.shape[0]

                pool: TensorType["datapoints","num_features"] = get_features(inputs, feature_size, model_wrapper)

                model_wrapper.model.eval()
                batch_size = self.params.aquisition_size
                batch_size = min(batch_size, N)

                if batch_size == 0:
                    self.current_aquisition += 1
                    return
                

                joint_entropy_class: GPCJointEntropy
                if True:
                    joint_entropy_class = LowMemMVNJointEntropy(model_wrapper.likelihood, samples, 10, N)
                if self.params.smoke_test:
                    joint_entropy_class_ = MVNJointEntropy(model_wrapper.likelihood, samples, 10, N)
                
                # We cant use the standard get_batchbald_batch function as we would need to sample and entire function from posterior
                # which is computationaly prohibative (has complexity related to the pool size)

                # We instead need to repeatedly compute the updated probabilties for each aquisition
                
                # We can instead of recomputing the entire distribtuion, we can compute all the pairs with the elements of the candidate batch
                # We can use this to build the new distributions for batch size
                # We will not directly manipulate the inducing points as there are various different strategies.
                # Instead we will we take advantage of the fact that GP output is a MVN and can be conditioned.

                features_expanded: TensorType["N", 1, "num_features"] = pool[:,None,:]
                ind_dists: MultitaskMultivariateNormalType[("N"), (1, "num_cats")] = get_gp_output(features_expanded, model_wrapper)
                conditional_entropies_N: TensorType["datapoints"] = compute_conditional_entropy_mvn(ind_dists, model_wrapper.likelihood, 100000).cpu()
                print("Cond")
                # print(conditional_entropies_N)

                # print(conditional_entropies_N)

                for i in tqdm(range(batch_size), desc="Aquiring", leave=False):
                    # First we compute the joint distribution of each of the datapoints with the current aquisition
                    # We first calculate the aquisition by itself first.

                    joint_entropy_result: TensorType["datapoints"] = torch.empty(N, dtype=torch.double, pin_memory=self.params.use_cuda)

                    # Add random sampling for larger aquisition sizes
                    # We can cache the size 2 distributions between the aquisitions (TODO)
                    # We can keep the current batch distribution from the prevoius aquisition (TODO)
                    # We perform rank-1 updates to the covariance and mean to get the new distributions 
                    # If we are performing variance reduction we could possible even make this cheaper

                    # Things to improve performance
                    # 1) caching of the feature tensors
                    # 2) don't recompute the distributions of things we have already calculated
                    # 3) Use much cleverer matrix ops on the join_rank_2 function
                    previous_aquisition: int = candidate_indices[-1] if i > 0 else 0 # When we don't have any candiates it doesn't matter
                    
                    expanded_pool_features: TensorType["datapoints", 1, 1, "num_features"] = pool[:, None, None, :]
                    new_candidate_features: TensorType["datapoints", 1, 1, "num_features"] = ((pool[previous_aquisition])[None, None, None, :]).expand(N, -1, -1, -1)
                    joint_features: TensorType["datapoints", 1, 2, "num_features"] = torch.cat([new_candidate_features, expanded_pool_features], dim=2)
                    dists: MultitaskMultivariateNormalType[ ("datapoints", 1), (2, "num_cat")] = get_gp_output(joint_features, model_wrapper)


                    rank2dist: Rank2Next = Rank2Next(dists)
                    if i > 0:
                        joint_entropy_class.add_variables(rank2dist, previous_aquisition) #type: ignore # last point
                        if self.params.smoke_test:
                           joint_entropy_class_.add_variables(rank2dist, previous_aquisition)
                    joint_entropy_result = joint_entropy_class.compute_batch(rank2dist)
                    if self.params.smoke_test:
                        joint_entropy_result_ = joint_entropy_class_.compute_batch(rank2dist)
                        difference = torch.flatten(joint_entropy_result_ - joint_entropy_result)
                        print(torch.std(difference))
                        print(torch.mean(difference))

                    shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

                    scores_N = joint_entropy_result.detach().cpu()
                    # scores_N[candidate_indices] = -float("inf")

                    scores_N -= conditional_entropies_N + shared_conditinal_entropies
                    scores_N[candidate_indices] = -float("inf")
                    # print(scores_N)

                    candidate_score, candidate_index = scores_N.max(dim=0)
                    
                    candidate_indices.append(candidate_index.item())
                    candidate_scores.append(candidate_score.item())

                    

                if use_bb_redux:
                    # We use the BatchBALD Redux as a comparision, this does not scale to larger pool sizes.
                    bb_samples = 1000
                    pool_expanded: TensorType[1, "datapoints", "num_features"] = pool[None,:,:]
                    # joint_distribution_list: MultitaskMultivariateNormalType[(1), ("datapoints", "num_cat")] = get_gp_output(pool_expanded, model_wrapper)
                    # assert(len(joint_distribution_list) == 1)
                    joint_distribution: MultitaskMultivariateNormalType = get_gp_output(pool_expanded, model_wrapper)
                    log_probs_N_K_C: TensorType["datapoints", "samples", "num_cat"] = ((model_wrapper.likelihood(joint_distribution.sample(sample_shape=torch.Size([bb_samples]))).logits).squeeze(1)).permute(1,0,2) # type: ignore
                    batch = get_batchbald_batch(log_probs_N_K_C, batch_size, 200000) 
                    redux_candidate_indices = batch.indices
                    redux_candidate_scores = batch.scores

                    print("Efficent")
                    print(candidate_indices)
                    print(candidate_scores)
                    for idx in candidate_indices:
                        _, y = dataset.get_pool_tensor()[idx]
                        print(y)

                    print("Redux")
                    print(redux_candidate_indices)
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
