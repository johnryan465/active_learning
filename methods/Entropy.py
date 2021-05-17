from uncertainty.multivariate_normal import MultitaskMultivariateNormalType
from uncertainty.estimator_entropy import CombinedJointEntropyEstimator, ExactJointEntropyEstimator, SampledJointEntropyEstimator, Sampling, get_entropy_batch
from uncertainty.bbredux_estimator_entropy import BBReduxJointEntropyEstimator

from utils.utils import get_pool
from models.model import UncertainModel
from models.vduq import vDUQ
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod
from methods.method_params import MethodParams, UncertainMethodParams
from batchbald_redux.batchbald import CandidateBatch, get_batchbald_batch

from typeguard import typechecked
from utils.typing import TensorType

import torch
from tqdm import tqdm
from dataclasses import dataclass
from toma import toma

from uncertainty.mvn_joint_entropy import CustomEntropy, GPCEntropy, Rank2Next


@dataclass
class EntropyParams(UncertainMethodParams):
    pass


class Entropy(UncertainMethod):
    def __init__(self, params: EntropyParams) -> None:
        super().__init__(params)

    @typechecked
    def score(self, model_wrapper: UncertainModel, inputs: TensorType) -> CandidateBatch:
        candidate_indices = []
        candidate_scores = []
        N = inputs.shape[0]
        num_cat = model_wrapper.get_num_cats()
        batch_size = self.params.aquisition_size
        batch_size = min(batch_size, N)
        if isinstance(model_wrapper, vDUQ):
            pool: TensorType["datapoints","num_features"] = model_wrapper.get_features(inputs)

            model_wrapper.model.eval()
            model_wrapper.likelihood.eval()

            features_expanded: TensorType["N", 1, "num_features"] = pool[:,None,:]
            ind_dists: MultitaskMultivariateNormalType = model_wrapper.get_gp_output(features_expanded)

            joint_entropy_class: GPCEntropy = CustomEntropy(model_wrapper.likelihood, self.params.samples, num_cat, N, ind_dists, CombinedJointEntropyEstimator)

            for i in tqdm(range(batch_size), desc="Aquiring", leave=False):
                previous_aquisition: int = candidate_indices[-1] if i > 0 else 0 # When we don't have any candiates it doesn't matter
                
                expanded_pool_features: TensorType["datapoints", 1, "num_features"] = pool[:, None, :]
                new_candidate_features: TensorType["datapoints", 1, "num_features"] = ((pool[previous_aquisition])[None, None, :]).expand(N, -1, -1)
                joint_features: TensorType["datapoints", 2, "num_features"] = torch.cat([new_candidate_features, expanded_pool_features], dim=1)
                dists: MultitaskMultivariateNormalType = model_wrapper.get_gp_output(joint_features)

                rank2dist: Rank2Next = Rank2Next(dists)
                if i > 0:
                    joint_entropy_class.add_variables(rank2dist, previous_aquisition)

                joint_entropy_result = joint_entropy_class.compute_batch(rank2dist)

                scores_N = joint_entropy_result.detach().clone().cpu()

                scores_N[candidate_indices] = -float("inf")

                candidate_score, candidate_index = scores_N.max(dim=0)
                
                candidate_indices.append(candidate_index.item())
                candidate_scores.append(candidate_score.item())

            return CandidateBatch(candidate_scores, candidate_indices)
        else:
            num_samples = self.params.samples.batch_samples
            samples = torch.zeros(N, num_samples, num_cat)
            model_wrapper.model.eval()
            @toma.execute.chunked(inputs, N)
            def make_samples(chunk: TensorType, start: int, end: int):
                if torch.cuda.is_available():
                    chunk = chunk.cuda()
                res = model_wrapper.sample(chunk, num_samples)
                samples[start:end].copy_(res)

            return get_entropy_batch(samples, batch_size, num_samples * self.params.samples.sum_samples)
