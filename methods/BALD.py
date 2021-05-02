from dataclasses import dataclass
from uncertainty.estimator_entropy import BBReduxJointEntropyEstimator
from uncertainty.mvn_joint_entropy import CustomJointEntropy
from typing import List
from utils.typing import MultitaskMultivariateNormalType

from models.model import UncertainModel
from models.vduq import vDUQ
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod, Method
from methods.method_params import MethodParams
from batchbald_redux.batchbald import get_bald_batch
from datasets.activelearningdataset import DatasetUtils
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
import torch
from .BatchBALD import get_pool, compute_conditional_entropy_mvn
from utils.typing import TensorType, MultitaskMultivariateNormalType, MultivariateNormalType

@dataclass
class BALDParams(MethodParams):
    samples: int


class BALD(UncertainMethod):
    def __init__(self, params: BALDParams) -> None:
        super().__init__()
        self.params = params
        self.current_aquisition = 0

    def acquire(self, model_wrapper: UncertainModel,
                dataset: ActiveLearningDataset, tb_logger: TensorboardLogger) -> None:
        if isinstance(model_wrapper, vDUQ):
            # We cant use the standard get_batchbald_batch function as we would need to sample and entire function from posterior
            # which is computationaly prohibative (has complexity related to the pool size)

            # We instead need to repeatedly compute the updated probabilties for each aquisition
            
            samples = 100
            num_cat = 10
            feature_size = 512

            inputs = get_pool(dataset)
            N = inputs.shape[0]

            pool = get_features(inputs, feature_size, model_wrapper)

            model_wrapper.model.eval()
            batch_size = self.params.aquisition_size
            batch_size = min(batch_size, N)

            if batch_size == 0:
                self.current_aquisition += 1
                return

            candidate_indices = []
            candidate_scores = []

            conditional_entropies_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())
            features_expanded: TensorType["datapoints", 1, "num_features"] = pool[:,None,:]
            ind_dists: List[MultitaskMultivariateNormalType[("chunk_size"), (1, "num_cats")]] = get_gp_output(features_expanded, model_wrapper)
            conditional_entropies_N = compute_conditional_entropy_mvn(ind_dists, model_wrapper.likelihood, samples).cpu()
            
            # First we compute the joint distribution of each of the datapoints with the current aquisition
            # We first calculate the aquisition by itself first.
            joint_entropy_result = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())
            scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

            # We get the current selected datapoints and broadcast them together with
            # the pool
            z = pool[candidate_indices]
            z = z[None,:,:]
            z = z.expand(N, -1, -1)

            t = pool[:,None,:]
            grouped_pool = torch.cat([z,t], dim=1)
            grouped_pool = grouped_pool[:,None,:,:]

            dists = get_gp_output(grouped_pool, model_wrapper)
            joint_entropy_result = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())
            joint_entropy_class = CustomJointEntropy(model_wrapper.likelihood, 60000, num_cat, N, ind_dists, BBReduxJointEntropyEstimator)

            joint_entropy_class.compute(dists, model_wrapper.likelihood, samples, joint_entropy_result)

            # Then we compute the batchbald objective

            shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

            scores_N = joint_entropy_result.detach().clone().cpu()

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
            for x, _ in dataset.get_pool():
                if torch.cuda.is_available():
                    x = x.cuda()
                probs_ = model_wrapper.sample(x, self.params.samples).detach().clone()
                probs.append(probs_)

            probs = torch.cat(probs, dim=0)
            batch = get_bald_batch(probs, self.params.aquisition_size)
            Method.log_batch(dataset.get_indexes(batch.indices), tb_logger, self.current_aquisition)
            dataset.move(batch.indices)
            self.current_aquisition += 1

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        DatasetUtils.balanced_init(dataset, self.params.initial_size)

    def complete(self) -> bool:
        return self.current_aquisition >= self.params.max_num_aquisitions
