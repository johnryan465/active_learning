
from datasets.dataset_params import DatasetParams
from methods.random import RandomParams
from models.model_params import GPParams, NNParams, OptimizerParams, TrainingParams
from models.vduq import vDUQParams


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