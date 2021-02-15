from params.params import Params
from marshmallow_dataclass import dataclass


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
    spectral_normalization: bool
    dropout_rate: float
    coeff: float
    n_power_iterations: int
    batchnorm_momentum: float
    weight_decay: float
