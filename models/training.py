from datasets.activelearningdataset import DatasetName
from dataclasses import dataclass
from .model_params import ModelParams
from typing import Optional

@dataclass
class OptimizerParams(ModelParams):
    optimizer: float
    var_optimizer: float = -1


@dataclass
class TrainingParams(ModelParams):
    epochs: int
    dataset: DatasetName
    cuda: bool
    optimizers: OptimizerParams
    batch_size: int
    patience: int
    progress_bar: bool
    objective: str = "nloss"
    profiler: bool = False
