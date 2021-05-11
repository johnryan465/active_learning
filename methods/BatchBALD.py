from uncertainty.multivariate_normal import MultitaskMultivariateNormalType
from uncertainty.estimator_entropy import ExactJointEntropyEstimator, SampledJointEntropyEstimator, Sampling
from uncertainty.bbredux_estimator_entropy import BBReduxJointEntropyEstimator

from utils.utils import get_pool
from datasets.activelearningdataset import DatasetUtils
from models.model import UncertainModel
from models.vduq import vDUQ
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod, Method
from methods.method_params import MethodParams
from batchbald_redux.batchbald import CandidateBatch, get_batchbald_batch
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
import torch.autograd.profiler as profiler

from typeguard import typechecked
from utils.typing import TensorType

import torch
from tqdm import tqdm
from dataclasses import dataclass
from toma import toma

from uncertainty.mvn_joint_entropy import CustomEntropy, GPCEntropy, Rank2Next



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
                ind_dists: MultitaskMultivariateNormalType = model_wrapper.get_gp_output(features_expanded)
                conditional_entropies_N: TensorType["datapoints"] = GPCEntropy.compute_conditional_entropy_mvn(ind_dists, model_wrapper.likelihood, 5000).cpu()

                joint_entropy_class: GPCEntropy = CustomEntropy(model_wrapper.likelihood, Sampling(batch_samples=300, per_samples=10, samples_sum=20), num_cat, N, ind_dists, SampledJointEntropyEstimator)
                if self.params.smoke_test:
                    joint_entropy_class_: GPCEntropy = CustomEntropy(model_wrapper.likelihood, Sampling(batch_samples=5000), num_cat, N, ind_dists, ExactJointEntropyEstimator)

                for i in tqdm(range(batch_size), desc="Aquiring", leave=False):
                    # First we compute the joint distribution of each of the datapoints with the current aquisition
                    # We first calculate the aquisition by itself first.

                    joint_entropy_result: TensorType["datapoints"] = torch.empty(N, dtype=torch.double, pin_memory=self.params.use_cuda)

                    previous_aquisition: int = candidate_indices[-1] if i > 0 else 0 # When we don't have any candiates it doesn't matter
                    
                    expanded_pool_features: TensorType["datapoints", 1, "num_features"] = pool[:, None, :]
                    new_candidate_features: TensorType["datapoints", 1, "num_features"] = ((pool[previous_aquisition])[None, None, :]).expand(N, -1, -1)
                    joint_features: TensorType["datapoints", 2, "num_features"] = torch.cat([new_candidate_features, expanded_pool_features], dim=1)
                    dists: MultitaskMultivariateNormalType = model_wrapper.get_gp_output(joint_features)

                    rank2dist: Rank2Next = Rank2Next(dists)
                    if i > 0:
                        joint_entropy_class.add_variables(rank2dist, previous_aquisition) #type: ignore # last point


                    # with profiler.profile(record_shapes=True) as prof:
                    #     with profiler.record_function("model_inference"):
                    #         joint_entropy_result = joint_entropy_class.compute_batch(rank2dist)
                    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=100))

                    import cProfile, pstats
                    profiler = cProfile.Profile()
                    profiler.enable()
                    joint_entropy_result = joint_entropy_class.compute_batch(rank2dist)
                    profiler.disable()
                    stats = pstats.Stats(profiler).sort_stats('cumtime')
                    stats.print_stats()

                    print(joint_entropy_result.shape)
                    print(joint_entropy_result)
                    
                    # print(joint_entropy_result)
                    if self.params.smoke_test:
                        if i > 0:
                            joint_entropy_class_.add_variables(rank2dist, previous_aquisition) #type: ignore # last point
                    


                    shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

                    scores_N = joint_entropy_result.detach().clone().cpu()

                    scores_N -= conditional_entropies_N + shared_conditinal_entropies
                    scores_N[candidate_indices] = -float("inf")

                    candidate_score, candidate_index = scores_N.max(dim=0)

                    if self.params.smoke_test:
                        joint_entropy_result_ = joint_entropy_class_.compute_batch(rank2dist) #type: ignore
                        # print(joint_entropy_result_)
                        diff = joint_entropy_result - joint_entropy_result_
                        # print(diff)
                        print(torch.mean(diff))
                        print(torch.std(diff))
                        pool_tensor = dataset.get_pool_tensor()
                        scores_N_ = joint_entropy_result_ - (conditional_entropies_N + shared_conditinal_entropies)
                        scores_N_[candidate_indices] = -float("inf")
                        candidate_score_, candidate_index_ = scores_N_.max(dim=0)

                        print("Sampled")
                        _, y = pool_tensor[candidate_index]
                        print(y, candidate_score)
                        print("BB Redux")
                        _, y_ = pool_tensor[candidate_index_]
                        print(y_, candidate_score_)

                        per_classes_idx = [ [] for i in range(num_cat)]

                        for idx in range(0, len(pool_tensor)):
                            _, y = pool_tensor[idx]
                            per_classes_idx[y].append(idx)

                        # The difference between the 2 methods is minimum at the max value of the low memory
                        difference = torch.flatten(diff)

                        for i in range(num_cat):
                            print("Class ", i)
                            class_diff = diff[per_classes_idx[i]]
                            difference = torch.flatten(class_diff)
                            print(torch.std(difference))
                            print(torch.mean(difference))

                    
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
                num_samples = 10
                samples = torch.zeros(N, num_samples, 10)
                @toma.execute.chunked(inputs, N)
                def make_samples(chunk: TensorType, start: int, end: int):
                    res = model_wrapper.sample(chunk, num_samples)
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
