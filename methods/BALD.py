from dataclasses import dataclass
from uncertainty.rank2 import Rank2Next
from uncertainty.estimator_entropy import Sampling
from uncertainty.multivariate_normal import MultitaskMultivariateNormalType
from uncertainty.bbredux_estimator_entropy import BBReduxJointEntropyEstimator
from uncertainty.mvn_joint_entropy import CustomEntropy, GPCEntropy
from typing import List

from models.model import UncertainModel
from models.vduq import vDUQ
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod, Method
from methods.method_params import MethodParams, UncertainMethodParams
from batchbald_redux.batchbald import CandidateBatch, get_bald_batch
from datasets.activelearningdataset import DatasetUtils
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
import torch
from .BatchBALD import get_pool
from utils.typing import TensorType

@dataclass
class BALDParams(UncertainMethodParams):
    pass


class BALD(UncertainMethod):
    def __init__(self, params: BALDParams) -> None:
        super().__init__(params)

    def score(self, model_wrapper: UncertainModel, inputs: TensorType) -> CandidateBatch:
        if isinstance(model_wrapper, vDUQ):
            num_cat = model_wrapper.get_num_cats()

            N = inputs.shape[0]

            pool = model_wrapper.get_features(inputs)

            model_wrapper.model.eval()
            batch_size = self.params.aquisition_size
            batch_size = min(batch_size, N)

            conditional_entropies_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())
            features_expanded: TensorType["datapoints", 1, "num_features"] = pool[:,None,:]
            ind_dists: MultitaskMultivariateNormalType = model_wrapper.get_gp_output(features_expanded)
            conditional_entropies_N = GPCEntropy.compute_conditional_entropy_mvn(ind_dists, model_wrapper.likelihood, self.params.samples.batch_samples).cpu()

            joint_entropy_class = CustomEntropy(model_wrapper.likelihood, self.params.samples, num_cat, N, ind_dists, BBReduxJointEntropyEstimator)

            new_candidate_features: TensorType["datapoints", 1, "num_features"] = ((pool[0])[None, None, :]).expand(N, -1, -1)
            joint_features: TensorType["datapoints", 2, "num_features"] = torch.cat([new_candidate_features, features_expanded], dim=1)
            dists: MultitaskMultivariateNormalType = model_wrapper.get_gp_output(joint_features)

            rank2dist: Rank2Next = Rank2Next(dists)
            joint_entropy_result = joint_entropy_class.compute_batch(rank2dist)

            scores_N = joint_entropy_result.cpu()

            scores_N -= conditional_entropies_N

            batch = torch.topk(scores_N, batch_size)

            return CandidateBatch(scores=batch.values.tolist(), indices=batch.indices.tolist())
        else:
            probs = []
            for x, _ in dataset.get_pool():
                if torch.cuda.is_available():
                    x = x.cuda()
                probs_ = model_wrapper.sample(x, self.params.samples).detach().clone()
                probs.append(probs_)

            probs = torch.cat(probs, dim=0)
            batch = get_bald_batch(probs, self.params.aquisition_size)
            return batch