from models.model_params import ModelParams
from datasets.dataset_params import DatasetParams
from methods.method_params import MethodParams
from params.params import Params

from dataclasses import dataclass


@dataclass
class ExperimentParams(Params):
    model_params : ModelParams
    method_params : MethodParams
    dataset_params : DatasetParams