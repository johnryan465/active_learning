from models.model_params import ModelParams
from models.training import TrainingParams
from datasets.dataset_params import DatasetParams
from methods.method_params import MethodParams
from params.params import Params

from dataclasses import dataclass


@dataclass
class ExperimentParams(Params):
    model_params: ModelParams
    training_params: TrainingParams
    method_params: MethodParams
    dataset_params: DatasetParams
