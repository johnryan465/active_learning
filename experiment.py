import json
from typing import Dict
from params.dataset_params import DatasetParams
from params.method_params import MethodParams
from experimental.experiment import Experiment
from models.driver import Driver
from models.bnn import BayesianMNIST
from models.dnn import DNN
from models.dnn import DNNParams
from models.vduq import vDUQ
from models.vduq import vDUQParams
from params.model_params import GPParams, ModelParams, NNParams, OptimizerParams, TrainingParams
from params.experiment_params import ExperimentParams
from methods.random import RandomParams
from methods.BALD import BALD
from datasets.activelearningdataset import DatasetName
from utils.config import IO

# from methods.BatchBALD import BatchBALD
import torch

import argparse

use_cuda = torch.cuda.is_available()

flag = False
bs = 256
epochs = 60



gp_params = GPParams(
    kernel = 'RBF',
    num_classes = 10,
    ard = None,
    n_inducing_points = 10,
    lengthscale_prior= False,
    separate_inducing_points = False
)

nn_params = NNParams(
    spectral_normalization = False,
    dropout_rate = 0.0,
    coeff = 0.9,
    n_power_iterations = 1,
    batchnorm_momentum = 0.01,
    weight_decay = 5e-4,
)

opt_params = OptimizerParams(
    optimizer = 0.01,
    var_optimizer = 0.01
)

training_params = TrainingParams(
    dataset = DatasetName.mnist,
    model_index = 0,
    batch_size = bs,
    epochs = epochs,
    cuda = use_cuda,
    optimizers = opt_params
)



def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [experiment_name]",
        description="Run experiments for active learning"
    )
    parser.add_argument('--name', required=True)
    return parser

if __name__ == "__main__":
    expr_config = ExperimentParams(
            model_params =  vDUQParams(
                training_params = training_params,
                fe_params = nn_params,
                gp_params = gp_params
            ),
            method_params = RandomParams(
                batch_size = bs
            ),
            dataset_params = DatasetParams(
                batch_size = bs
            )
    )
    s = expr_config.export()
    e = IO.parseParams(ExperimentParams, s)
    expr = Experiment(
        "Rerun",
        "Testing Experimental Framework",
        expr_config
    )
    # expr.run()
