from datasets.activelearningdataset import DatasetName
from marshmallow_dataclass import dataclass
from .model_params import ModelParams


@dataclass
class OptimizerParams(ModelParams):
    optimizer: float
    var_optimizer: float = None


@dataclass
class TrainingParams(ModelParams):
    epochs: int
    dataset: DatasetName
    cuda: bool
    optimizers: OptimizerParams
    batch_size: int
    patience: int
    progress_bar: bool
