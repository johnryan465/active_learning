from torch.tensor import Tensor
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
from batchbald_redux.batchbald import CandidateBatch, compute_conditional_entropy, get_batchbald_batch
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
            candidate_indices = []
            candidate_scores = []
            inputs: TensorType["datapoints","channels","x","y"] = get_pool(dataset)
            N = inputs.shape[0]
            num_cat = 10
            batch_size = self.params.aquisition_size
            batch_size = min(batch_size, N)
            if isinstance(model_wrapper, vDUQ):
                pool: TensorType["datapoints","num_features"] = model_wrapper.get_features(inputs)

                model_wrapper.model.eval()
                
                
                # We cant use the standard get_batchbald_batch function as we would need to sample and entire function from posterior
                # which is computationaly prohibative (has complexity cubically related to the pool size)

                # We instead need to repeatedly compute the updated probabilties for each aquisition
                
                # We can instead of recomputing the entire distribtuion, we can compute all the pairs with the elements of the candidate batch
                # We can use this to build the new distributions for batch size
                # We will not directly manipulate the inducing points as there are various different strategies.
                # Instead we will we take advantage of the fact that GP output is a MVN and can be conditioned.

                features_expanded: TensorType["N", 1, "num_features"] = pool[:,None,:]
                ind_dists: MultitaskMultivariateNormalType[("N"), (1, "num_cats")] = model_wrapper.get_gp_output(features_expanded)
                conditional_entropies_N: TensorType["datapoints"] = compute_conditional_entropy_mvn(ind_dists, model_wrapper.likelihood, 5000).cpu()

                joint_entropy_class: GPCJointEntropy = CustomJointEntropy(model_wrapper.likelihood, 60000, num_cat, N, ind_dists, SampledJointEntropyEstimator)
                # joint_entropy_class: GPCJointEntropy = CustomJointEntropy(model_wrapper.likelihood, 1500, num_cat, N, ind_dists, ExactJointEntropyEstimator)

                for i in tqdm(range(batch_size), desc="Aquiring", leave=False):
                    # First we compute the joint distribution of each of the datapoints with the current aquisition
                    # We first calculate the aquisition by itself first.

                    joint_entropy_result: TensorType["datapoints"] = torch.empty(N, dtype=torch.double, pin_memory=self.params.use_cuda)

                    previous_aquisition: int = candidate_indices[-1] if i > 0 else 0 # When we don't have any candiates it doesn't matter
                    
                    expanded_pool_features: TensorType["datapoints", 1, 1, "num_features"] = pool[:, None, None, :]
                    new_candidate_features: TensorType["datapoints", 1, 1, "num_features"] = ((pool[previous_aquisition])[None, None, None, :]).expand(N, -1, -1, -1)
                    joint_features: TensorType["datapoints", 1, 2, "num_features"] = torch.cat([new_candidate_features, expanded_pool_features], dim=2)
                    dists: MultitaskMultivariateNormalType[ ("datapoints", 1), (2, "num_cat")] = model_wrapper.get_gp_output(joint_features)

                    rank2dist: Rank2Next = Rank2Next(dists)
                    if i > 0:
                        joint_entropy_class.add_variables(rank2dist, previous_aquisition) #type: ignore # last point

                    joint_entropy_result = joint_entropy_class.compute_batch(rank2dist)
                    print(joint_entropy_result)

                    shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

                    scores_N = joint_entropy_result.detach().cpu()

                    # scores_N -= conditional_entropies_N + shared_conditinal_entropies
                    scores_N[candidate_indices] = -float("inf")

                    candidate_score, candidate_index = scores_N.max(dim=0)
                    
                    candidate_indices.append(candidate_index.item())
                    candidate_scores.append(candidate_score.item())

                batch = CandidateBatch(candidate_scores, candidate_indices)
                print(batch)

                if self.params.smoke_test:
                    # We use the BatchBALD Redux as a comparision, this does not scale to larger pool sizes.
                    bb_samples = 5000
                    pool_expanded: TensorType[1, "datapoints", "num_features"] = pool[None,:,:]

                    joint_distribution: MultitaskMultivariateNormalType = model_wrapper.get_gp_output(pool_expanded)
                    log_probs_N_K_C: TensorType["datapoints", "samples", "num_cat"] = ((model_wrapper.likelihood(joint_distribution.sample(sample_shape=torch.Size([bb_samples]))).logits).squeeze(1)).permute(1,0,2) # type: ignore
                    batch_ = get_batchbald_batch(log_probs_N_K_C, batch_size, 600000) 
                    print(batch_)


            else:
                num_samples = 1000
                samples = torch.zeros(N, num_samples, 10)
                @toma.execute.chunked(inputs, N)
                def make_samples(chunk: TensorType, start: int, end: int):
                    res = model_wrapper.sample(chunk, 1000)
                    samples[start:end].copy_(res)

                batch = get_batchbald_batch(samples, batch_size, 60000)
                candidate_indices = batch.indices
                candidate_scores = batch.scores
            Method.log_batch(dataset.get_indexes(candidate_indices), tb_logger, self.current_aquisition)
            dataset.move(candidate_indices)

            self.current_aquisition += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    def initialise(self, dataset: ActiveLearningDataset) -> None:
        DatasetUtils.balanced_init(dataset, self.params.initial_size)

    def complete(self) -> bool:
        return self.current_aquisition >= self.params.max_num_aquisitions
