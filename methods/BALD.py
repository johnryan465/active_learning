from dataclasses import dataclass

from models.model import UncertainModel
from models.vduq import vDUQ
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod, Method
from methods.method_params import MethodParams
from batchbald_redux.batchbald import get_bald_batch
from datasets.activelearningdataset import DatasetUtils
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
import torch
from .BatchBALD import get_pool, get_features, get_ind_output, compute_conditional_entropy_mvn, joint_entropy_mvn, get_gp_output

@dataclass
class BALDParams(MethodParams):
    samples: int


class BALD(UncertainMethod):
    def __init__(self, params: BALDParams) -> None:
        super().__init__()
        self.params = params
        self.current_aquisition = 0

    def acquire(self, model: UncertainModel,
                dataset: ActiveLearningDataset, tb_logger: TensorboardLogger) -> None:
        if isinstance(model, vDUQ):
            # We cant use the standard get_batchbald_batch function as we would need to sample and entire function from posterior
            # which is computationaly prohibative (has complexity related to the pool size)

            # We instead need to repeatedly compute the updated probabilties for each aquisition
            
            samples = 100
            num_cat = 10
            feature_size = 512

            inputs = get_pool(dataset)
            N = inputs.shape[0]

            pool = get_features(inputs, feature_size, model)

            model.model.eval()
            batch_size = self.params.aquisition_size
            batch_size = min(batch_size, N)

            if batch_size == 0:
                self.current_aquisition += 1
                return

            candidate_indices = []
            candidate_scores = []

            conditional_entropies_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

            dists = get_ind_output(pool, model)
            conditional_entropies_N = compute_conditional_entropy_mvn(dists, model.likelihood, samples).cpu()
            
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

            dists = get_gp_output(grouped_pool, model)

            joint_entropy_result = joint_entropy_mvn(dists, model.likelihood, samples, num_cat)

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
                probs_ = model.sample(x, self.params.samples).detach().clone()
                probs.append(probs_)

            probs = torch.cat(probs, dim=0)
            batch = get_bald_batch(probs, self.params.aquisition_size)
            Method.log_batch(batch.indices, tb_logger, self.current_aquisition)
            dataset.move(batch.indices)
            self.current_aquisition += 1

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        DatasetUtils.balanced_init(dataset, self.params.initial_size)

    def complete(self) -> bool:
        return self.current_aquisition >= self.params.max_num_aquisitions
