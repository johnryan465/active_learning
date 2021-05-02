from datasets.activelearningdataset import ActiveLearningDataset
from utils.typing import MultitaskMultivariateNormalType, MultivariateNormalType, TensorType
from typeguard import check_type, typechecked
from tqdm import tqdm
import torch

# Gets the pool from the dataset as a tensor
@typechecked
def get_pool(dataset: ActiveLearningDataset) -> TensorType["datapoints", "channels", "x", "y"]:
    inputs = []
    for x, _ in tqdm(dataset.get_pool_tensor(), desc="Loading pool", leave=False):
        inputs.append(x)
    inputs = torch.stack(inputs, dim=0)
    return inputs
