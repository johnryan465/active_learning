
from methods.BatchBALD import BatchBALDParams
from models.bnn import BNNParams
from methods.BALD import BALDParams
from datasets.dataset_params import DatasetParams
from methods.random import RandomParams
from models.model_params import GPParams, NNParams
from models.training import OptimizerParams, TrainingParams
from models.vduq import vDUQParams


class Registered:
    types = {
        'ModelParams': [vDUQParams, NNParams, GPParams, BNNParams],
        'TrainingParams': [TrainingParams],
        'OptimizerParams': [OptimizerParams],
        'NNParams': [NNParams],
        'GPParams': [GPParams],
        'MethodParams': [RandomParams, BALDParams, BatchBALDParams],
        'DatasetParams': [DatasetParams]
    }
