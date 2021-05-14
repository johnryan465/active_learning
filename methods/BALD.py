from dataclasses import dataclass
from methods.method import UncertainMethod
from uncertainty.rank2 import Rank2Next
from uncertainty.estimator_entropy import CombinedJointEntropyEstimator, SampledJointEntropyEstimator
from uncertainty.multivariate_normal import MultitaskMultivariateNormalType
from uncertainty.mvn_joint_entropy import CustomEntropy, GPCEntropy

from models.model import UncertainModel
from models.vduq import vDUQ
from methods.method_params import UncertainMethodParams
from batchbald_redux.batchbald import CandidateBatch, get_bald_batch
import torch
from toma import toma
from utils.typing import TensorType

@dataclass
class BALDParams(UncertainMethodParams):
    pass


class BALD(UncertainMethod):
    def __init__(self, params: BALDParams) -> None:
        super().__init__(params)

    def score(self, model_wrapper: UncertainModel, inputs: TensorType) -> CandidateBatch:
        num_cat = model_wrapper.get_num_cats()
        N = inputs.shape[0]
        batch_size = self.params.aquisition_size
        batch_size = min(batch_size, N)
        if isinstance(model_wrapper, vDUQ):
            pool = model_wrapper.get_features(inputs)

            model_wrapper.model.eval()
            model_wrapper.likelihood.eval()

            conditional_entropies_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())
            features_expanded: TensorType["datapoints", 1, "num_features"] = pool[:,None,:]
            ind_dists: MultitaskMultivariateNormalType = model_wrapper.get_gp_output(features_expanded)
            conditional_entropies_N = GPCEntropy.compute_conditional_entropy_mvn(ind_dists, model_wrapper.likelihood, self.params.samples.batch_samples).cpu()

            joint_entropy_class = CustomEntropy(model_wrapper.likelihood, self.params.samples, num_cat, N, ind_dists, CombinedJointEntropyEstimator)

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
            num_samples = self.params.samples.batch_samples
            samples = torch.zeros(N, num_samples, num_cat)
            @toma.execute.chunked(inputs, N)
            def make_samples(chunk: TensorType, start: int, end: int):
                res = model_wrapper.sample(chunk, num_samples)
                samples[start:end].copy_(res)

            return get_bald_batch(samples, batch_size, num_samples * self.params.samples.sum_samples)