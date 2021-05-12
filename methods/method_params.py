from uncertainty.estimator_entropy import Sampling
from params.params import Params
from dataclasses import dataclass


@dataclass
class MethodParams(Params):
    aquisition_size: int
    max_num_aquisitions: int
    initial_size: int
    smoke_test: bool

@dataclass
class UncertainMethodParams(MethodParams):
    samples: Sampling