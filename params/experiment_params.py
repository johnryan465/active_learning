from abc import ABC, abstractmethod
from params.model_params import ModelParams
from params.dataset_params import DatasetParams
from params.method_params import MethodParams
from params.params import Params

from dataclasses import dataclass


@dataclass
class ExperimentParams(Params):
    model_params : ModelParams
    method_params : MethodParams
    dataset_params : DatasetParams