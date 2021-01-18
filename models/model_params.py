from abc import ABC, abstractmethod

from marshmallow_oneofschema.one_of_schema import OneOfSchema
from params.params import Params
from typing import List, Union
from datasets.activelearningdataset import DatasetName
from marshmallow_dataclass import dataclass
from marshmallow.validate import Equal
from dataclasses import field

@dataclass
class ModelParams(Params):
    pass

@dataclass
class GPParams(ModelParams):
    n_inducing_points : int
    num_classes : int
    separate_inducing_points : bool
    kernel : str
    ard : int
    lengthscale_prior : bool

@dataclass
class NNParams(ModelParams):
    spectral_normalization : bool
    dropout_rate : float
    coeff : float
    n_power_iterations : int
    batchnorm_momentum : float
    weight_decay : float

@dataclass
class OptimizerParams(ModelParams):
    optimizer : float
    var_optimizer : float = None

@dataclass
class TrainingParams(ModelParams):
    epochs : int
    dataset : DatasetName
    cuda : bool
    optimizers : OptimizerParams
    batch_size : int
    model_index : int = 0

