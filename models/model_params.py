from params.params import Params
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelParams(Params):
    def toDict(self) -> dict:
        return self.__dict__


@dataclass
class ModelWrapperParams(ModelParams):
    model_index: int


@dataclass
class GPParams(ModelParams):
    n_inducing_points: int
    num_classes: int
    separate_inducing_points: bool
    kernel: str
    ard: int
    lengthscale_prior: bool
    distribution: str


@dataclass
class NNParams(ModelParams):
    batchnorm_momentum: float
    weight_decay: float
    dropout_rate: float
    spectral_normalization: bool = False
    coeff: float = 9.0
    n_power_iterations: int = 1
    num_classes: int = None # type: ignore

@dataclass
class DNNParams(ModelWrapperParams):
    nn_params: NNParams

