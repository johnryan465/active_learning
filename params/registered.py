
from params.dataset_params import DatasetParams
from methods.random import RandomParams
from params.model_params import GPParams, NNParams, OptimizerParams, TrainingParams, vDUQParams


class Registered:
    types = {
        'ModelParams' : [vDUQParams, NNParams, GPParams],
        'TrainingParams' : [TrainingParams],
        'OptimizerParams' : [OptimizerParams],
        'NNParams' : [NNParams],
        'GPParams' : [GPParams],
        'MethodParams' : [RandomParams],
        'DatasetParams' : [DatasetParams]
    }