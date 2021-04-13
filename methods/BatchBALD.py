from torch.distributions.normal import Normal
from datasets.activelearningdataset import DatasetUtils
from models.model import UncertainModel
from models.vduq import vDUQ
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod, Method
from methods.method_params import MethodParams
from batchbald_redux.batchbald import get_batchbald_batch, CandidateBatch
from batchbald_redux import joint_entropy
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

from gpytorch.lazy import CatLazyTensor, BlockDiagLazyTensor
from torch.distributions.utils import _standard_normal
from typing import Annotated, Callable, Generic, List, NewType, Tuple, Type, TypeVar, Union
from typeguard import typechecked
from utils.typing import MultitaskMultivariateNormalType, MultivariateNormalType, TensorType

import torch
from tqdm import tqdm
from dataclasses import dataclass
from toma import toma
import string



@dataclass
class BatchBALDParams(MethodParams):
    samples: int
    use_cuda: bool
    var_reduction: bool = True




# \sigma_{BatchBALD} ( {x_1, ..., x_n}, p(w)) = H(y_1, ..., y_n) - E_{p(w)} H(y | x, w)
@typechecked
def combine_mtmvns(mvns) -> MultitaskMultivariateNormalType:
    if len(mvns) < 2:
        return mvns[0]

    if not all(m.event_shape == mvns[0].event_shape for m in mvns[1:]):
        raise ValueError("All MultivariateNormals must have the same event shape")
    mean = torch.cat([mvn.mean for mvn in mvns], 0)

    covar_blocks_lazy = CatLazyTensor(
        *[mvn.lazy_covariance_matrix for mvn in mvns], dim=0, output_device=mean.device
    )
    covar_lazy = BlockDiagLazyTensor(covar_blocks_lazy, block_dim=0)
    return MultitaskMultivariateNormalType(mean=mean, covariance_matrix=covar_blocks_lazy, interleaved=False)

# We compute the joint entropy of the distributions
@typechecked
def joint_entropy_mvn(distributions: List[MultivariateNormalType], likelihood, per_samples, num_configs: int, output: torch.Tensor, variance_reduction: bool = False) -> None:
    # We can exactly compute a larger sized exact distribution
    # As the task batches are independent we can chunk them
    D = distributions[0].event_shape[0]
    N = sum([distribution.batch_shape[0] for distribution in distributions])
    t = string.ascii_lowercase[:D]
    s =  ','.join(['yz' + c for c in list(t)]) + '->' + 'yz' + t

    if D < 5:
        # We can chunk to get away with lower memory usage
        # We can't lazily split, only lazily combine
        len_d = len(distributions)
        if variance_reduction:
            # here is where we generate the samples which we will 
            shape = [per_samples] + list(distributions[0].base_sample_shape)
            samples = _standard_normal(torch.Size(shape), dtype=distributions[0].loc.dtype, device=distributions[0].loc.device)

        def func(distribution: MultivariateNormalType) -> TensorType:
            if variance_reduction:
                base_samples = samples.detach().clone()
                base_samples = base_samples[:,None,:,:]
                base_samples = base_samples.expand(-1, distribution.batch_shape[0], -1, -1)
                l = likelihood(distribution.sample(base_samples=base_samples)).probs
            else:
                l = likelihood(distribution.sample(sample_shape=torch.Size([per_samples]))).probs
            l = torch.transpose(l, 0, 1)
            g = torch.einsum(s, *torch.unbind(l, dim=2))
            g = g * torch.log(g)
            g = -torch.sum(g.flatten(2), 1) / per_samples
            return g
        
        chunked_distribution("Joint Entropy", distributions, func, output)
    else:
        return None

@typechecked
def chunked_distribution(name: str, distributions: List[MultivariateNormalType], func: Callable, output: TensorType["N": ...]) -> None:
    N = output.shape[0]
    len_d = len(distributions)
    @toma.batch(initial_batchsize=len_d)
    def compute(batchsize: int, distributions: List[MultivariateNormalType]):
        pbar = tqdm(total=N, desc="name", leave=False)
        start = 0
        end = 0
        for i in range(0, len_d, batchsize):
            distribution = combine_mtmvns(distributions[i: min(i+batchsize, len_d)])
            end = start + distribution.batch_shape[0]
            g = func(distribution)
            output[start:end].copy_(g)
            pbar.update(end - start)
            start = end
            pbar.close()
    compute(distributions) #type: ignore

# Gets the pool from the dataset as a tensor
@typechecked
def get_pool(dataset: ActiveLearningDataset) -> TensorType["datapoints", "channels", "x", "y"]:
    inputs = []
    for x, _ in tqdm(dataset.get_pool_tensor(), desc="Loading pool", leave=False):
        inputs.append(x)
    inputs = torch.stack(inputs, dim=0)
    return inputs


@typechecked
def get_features(inputs: TensorType["datapoints", "channels", "x", "y"], feature_size: int, model_wrapper: vDUQ) -> TensorType["datapoints", "num_features"]:
    N = inputs.shape[0]
    pool = torch.empty((N, feature_size))
    pbar = tqdm(total=N, desc="Feature Extraction", leave=False)
    @toma.execute.chunked(inputs, 1024)
    def compute(inputs, start: int, end: int):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            tmp = model_wrapper.model.feature_extractor.forward(inputs).detach()
            pool[start:end].copy_(tmp)
        pbar.update(end - start)
    pbar.close()
    return pool


@typechecked
def get_gp_output(features: TensorType["datapoints", "num_points", "num_features"], model_wrapper: vDUQ) -> List[MultivariateNormalType[("chunk_size"), (1, "num_cats")]]:
    features = features[:,None,:,:]
    with torch.no_grad():
        dists = []
        N = features.shape[0]
        pbar = tqdm(total=N, desc="GP", leave=False)
        @toma.execute.chunked(features, 256)
        def compute(features, start: int, end: int):
            if torch.cuda.is_available():
                features = features.cuda()
            d = model_wrapper.model.gp(features)
            dists.append(d)
            pbar.update(end - start)
        pbar.close()
        return dists


@typechecked
def compute_conditional_entropy_mvn(distributions: List[MultitaskMultivariateNormalType[("chunk_size"), (1, "num_cats")]], likelihood, num_samples : int) -> TensorType["datapoints"]:
    # The distribution input is a batch of MVNS
    N = sum(map(lambda x: x.batch_shape[0], distributions))
    log_probs_N_K_C = torch.empty(N, num_samples, 10)
    def func(distribution: MultitaskMultivariateNormalType) -> TensorType:
        h = likelihood(distribution.sample(sample_shape=torch.Size([num_samples]))).logits
        h = h.permute(1, 0, 3, 2).squeeze(3)
        return h
    chunked_distribution("Sampling", distributions, func, log_probs_N_K_C)
    N, K, C = log_probs_N_K_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)
    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

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
                # We cant use the standard get_batchbald_batch function as we would need to sample and entire function from posterior
                # which is computationaly prohibative (has complexity related to the pool size)

                # We instead need to repeatedly compute the updated probabilties for each aquisition
                
                samples = self.params.samples
                num_cat = 10
                feature_size = 512

                inputs: TensorType["datapoints","channels","x","y"] = get_pool(dataset)
                N = inputs.shape[0]

                pool: TensorType["datapoints","num_features"] = get_features(inputs, feature_size, model_wrapper)

                model_wrapper.model.eval()
                batch_size = self.params.aquisition_size
                batch_size = min(batch_size, N)

                if batch_size == 0:
                    self.current_aquisition += 1
                    return

                candidate_indices = []
                candidate_scores = []

                # Todo, creation this large distribution is unnecessary so do what we do for joint
                features_expanded: TensorType["datapoints", 1, "num_features"] = pool[:,None,:]
                ind_dists: List[MultitaskMultivariateNormalType[("chunk_size"), (1, "num_cats")]] = get_gp_output(features_expanded, model_wrapper)
                conditional_entropies_N: TensorType["datapoints"] = compute_conditional_entropy_mvn(ind_dists, model_wrapper.likelihood, samples).cpu()
                
                for i in tqdm(range(batch_size), desc="Acquiring", leave=False):
                    # First we compute the joint distribution of each of the datapoints with the current aquisition
                    # We first calculate the aquisition by itself first.

                    joint_entropy_result: TensorType["datapoints"] = torch.empty(N, dtype=torch.double, pin_memory=self.params.use_cuda)

                    # We get the current selected datapoints and broadcast them together with
                    # the pool
                    z: TensorType["current_batch_size", "num_features"] = pool[candidate_indices]
                    z: TensorType[1, "current_batch_size", "num_features"]= z[None,:,:]
                    z: TensorType["datapoints", "current_batch_size", "num_features"] = z.expand(N, -1, -1)

                    t: TensorType["datapoints", 1, "num_features"] = pool[:,None,:]
                    grouped_pool: TensorType["datapoints": "new_batch_size", "num_features"] = torch.cat([z,t], dim=1)

                    dists: List[MultitaskMultivariateNormalType[("chunked"), ("new_batch_size", "num_cat")]] = get_gp_output(grouped_pool, model_wrapper)

                    joint_entropy_mvn(dists, model_wrapper.likelihood, samples, num_cat, joint_entropy_result, variance_reduction=self.params.var_reduction)

                    # Then we compute the batchbald objective

                    shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

                    scores_N = joint_entropy_result.detach().cpu()

                    scores_N -= conditional_entropies_N + shared_conditinal_entropies
                    scores_N[candidate_indices] = -float("inf")

                    candidate_score, candidate_index = scores_N.max(dim=0)
                    
                    candidate_indices.append(candidate_index.item())
                    candidate_scores.append(candidate_score.item())
                
                Method.log_batch(dataset.get_indexes(candidate_indices), tb_logger, self.current_aquisition)
                dataset.move(candidate_indices)

                self.current_aquisition += 1

            else:
                raise NotImplementedError("BatchBALD")

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        DatasetUtils.balanced_init(dataset, self.params.initial_size)

    def complete(self) -> bool:
        return self.current_aquisition >= self.params.max_num_aquisitions
