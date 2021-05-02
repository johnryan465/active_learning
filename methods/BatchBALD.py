from uncertainty.estimator_entropy import BBReduxJointEntropyEstimator, ExactJointEntropyEstimator, SampledJointEntropyEstimator
from uncertainty.mvn_utils import combine_mtmvns
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from gpytorch.lazy.cat_lazy_tensor import CatLazyTensor
from gpytorch.likelihoods import likelihood
from utils.utils import get_pool
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

from uncertainty.mvn_joint_entropy import CustomJointEntropy, GPCJointEntropy, Rank2Next, chunked_distribution, compute_conditional_entropy_mvn



@dataclass
class BatchBALDParams(MethodParams):
    samples: int
    use_cuda: bool
    var_reduction: bool = True
    efficent: bool = True


# \sigma_{BatchBALD} ( {x_1, ..., x_n}, p(w)) = H(y_1, ..., y_n) - E_{p(w)} H(y | x, w)


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
                use_bb_redux = self.params.smoke_test

                inputs: TensorType["datapoints","channels","x","y"] = get_pool(dataset)
                N = inputs.shape[0]

                pool: TensorType["datapoints","num_features"] = model_wrapper.get_features(inputs)

                model_wrapper.model.eval()
                batch_size = self.params.aquisition_size
                batch_size = min(batch_size, N)

                if batch_size == 0:
                    self.current_aquisition += 1
                    return
                

                
                # We cant use the standard get_batchbald_batch function as we would need to sample and entire function from posterior
                # which is computationaly prohibative (has complexity related to the pool size)

                # We instead need to repeatedly compute the updated probabilties for each aquisition
                
                # We can instead of recomputing the entire distribtuion, we can compute all the pairs with the elements of the candidate batch
                # We can use this to build the new distributions for batch size
                # We will not directly manipulate the inducing points as there are various different strategies.
                # Instead we will we take advantage of the fact that GP output is a MVN and can be conditioned.

                features_expanded: TensorType["N", 1, "num_features"] = pool[:,None,:]
                ind_dists: MultitaskMultivariateNormalType[("N"), (1, "num_cats")] = model_wrapper.get_gp_output(features_expanded)
                conditional_entropies_N: TensorType["datapoints"] = compute_conditional_entropy_mvn(ind_dists, model_wrapper.likelihood, 5000).cpu()
                print("Cond")

                joint_entropy_class: GPCJointEntropy
                if True:
                    # joint_entropy_class = LowMemMVNJointEntropy(model_wrapper.likelihood, 10, 1000, 1, num_cat, N)
                    joint_entropy_class = CustomJointEntropy(model_wrapper.likelihood, 60000, num_cat, N, ind_dists, SampledJointEntropyEstimator)
                    joint_entropy_class_ = CustomJointEntropy(model_wrapper.likelihood, 60000, num_cat, N, ind_dists, ExactJointEntropyEstimator)
                    # joint_entropy_class = MVNJointEntropy(model_wrapper.likelihood, 50, 10, N)
                if self.params.smoke_test:
                    pass
                    # joint_entropy_class_ = MVNJointEntropy(model_wrapper.likelihood, 1000, num_cat, N)

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
                    dists: MultitaskMultivariateNormalType[ ("datapoints", 1), (2, "num_cat")] = model_wrapper.get_gp_output(joint_features)


                    rank2dist: Rank2Next = Rank2Next(dists)
                    if i > 0:
                        joint_entropy_class.add_variables(rank2dist, previous_aquisition) #type: ignore # last point
                        if self.params.smoke_test:
                           joint_entropy_class_.add_variables(rank2dist, previous_aquisition)
                    joint_entropy_result = joint_entropy_class.compute_batch(rank2dist)
                    if self.params.smoke_test:
                        expanded_pool_features: TensorType["datapoints", 1, "num_features"] = pool[:, None, :]
                        new_candidate_features: TensorType["datapoints", 1, "num_features"] = ((pool[candidate_indices])[None, :, :]).expand(N, -1, -1)
                        joint_features: TensorType["datapoints", "new_batch_size", "num_features"] = torch.cat([new_candidate_features, expanded_pool_features], dim=1)
                        new_dist = model_wrapper.get_gp_output(joint_features)

                        # Here we check that the distribuiton we get from combining 
                        # check_equal_dist(joint_entropy_class_.join_rank_2(rank2dist), new_dist)
                        # joint_entropy_result_ = joint_entropy_class_.compute_batch(rank2dist)
                        # exact, sampled = MVNJointEntropy._compute(new_dist, model_wrapper.likelihood, 50000, 10)
                        # exact2, sampled2 = MVNJointEntropy._compute(new_dist, model_wrapper.likelihood, 50000, 10)
                        joint_entropy_result_ = joint_entropy_class_.compute_batch(rank2dist)


                        
                        # print("Exact", exact)
                        # print(joint_entropy_result)
                        # sampled = joint_entropy_result
                        # joint_entropy_result = exact
                        # joint_entropy_result = exact
                        # print("Simple")
                        # print(simple_joint_entropy_result)
                        # print("Low Memory")
                        # print(joint_entropy_result)
                        # # print("Rank 2 combine only")
                        # # print(joint_entropy_result_)

                        pool_tensor = dataset.get_pool_tensor()

                        diff_1 = joint_entropy_result - joint_entropy_result_
                        diff_1[candidate_indices] = 0

                        print(diff_1)
                        print(joint_entropy_result_)



                        per_classes_idx = [ [] for i in range(num_cat)]

                        for idx in range(0, len(pool_tensor)):
                            _, y = pool_tensor[idx]
                            per_classes_idx[y].append(idx)

                        # The difference between the 2 methods is minimum at the max value of the low memory
                        difference = torch.flatten(diff_1)

                        for i in range(num_cat):
                            print("Class ", i)
                            class_diff = diff_1[per_classes_idx[i]]
                            difference = torch.flatten(class_diff)
                            print(torch.std(difference))
                            print(torch.mean(difference))
                        # print(torch.mean(difference))
                        # print(joint_entropy_result)
                        # print(sampled)
                        # print(exact)
                        # indexes = torch.argsort(difference)
                        # print("Scores")
                        # for idx in indexes:
                        #     _, y = pool_tensor[idx]
                        #     print(idx, difference[idx], y)

                        # print(torch.std(difference))
                        # print(torch.mean(difference))

                    shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

                    scores_N = joint_entropy_result.detach().cpu()

                    # scores_N -= conditional_entropies_N + shared_conditinal_entropies
                    scores_N[candidate_indices] = -float("inf")

                    # print(scores_N)

                    candidate_score, candidate_index = scores_N.max(dim=0)
                    
                    candidate_indices.append(candidate_index.item())
                    candidate_scores.append(candidate_score.item())

                    pool_tensor = dataset.get_pool_tensor()
                    print("Scores")
                    for idx in candidate_indices:
                        _, y = pool_tensor[idx]
                        print(idx, scores_N[idx], y)

                    

                if use_bb_redux:
                    # We use the BatchBALD Redux as a comparision, this does not scale to larger pool sizes.
                    bb_samples = 5000
                    pool_expanded: TensorType[1, "datapoints", "num_features"] = pool[None,:,:]
                    # joint_distribution_list: MultitaskMultivariateNormalType[(1), ("datapoints", "num_cat")] = get_gp_output(pool_expanded, model_wrapper)
                    # assert(len(joint_distribution_list) == 1)
                    joint_distribution: MultitaskMultivariateNormalType = model_wrapper.get_gp_output(pool_expanded)
                    log_probs_N_K_C: TensorType["datapoints", "samples", "num_cat"] = ((model_wrapper.likelihood(joint_distribution.sample(sample_shape=torch.Size([bb_samples]))).logits).squeeze(1)).permute(1,0,2) # type: ignore
                    log_probs_N_K_C_: TensorType["datapoints", "samples", "num_cat"] = ((model_wrapper.likelihood(joint_distribution.sample(sample_shape=torch.Size([bb_samples]))).logits).squeeze(1)).permute(1,0,2) # type: ignore
                    batch_ = get_batchbald_batch(log_probs_N_K_C_, batch_size, 600000)
                    batch = get_batchbald_batch(log_probs_N_K_C, batch_size, 600000) 
                    print(batch_)
                    print(batch)
                    redux_candidate_indices = batch.indices
                    redux_candidate_scores = batch.scores

                    # print("Efficent")
                    # print(candidate_indices)
                    # print(candidate_scores)
                    # for idx in candidate_indices:
                    #     _, y = dataset.get_pool_tensor()[idx]
                    #     print(y)

                    print("Noisy Redux")
                    for idx in batch_.indices: # type: ignore
                        _, y = dataset.get_pool_tensor()[idx] # type: ignore
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
