import json
from typing import Dict
from datasets.dataset_params import DatasetParams
from methods.method_params import MethodParams
from experimental.experiment import Experiment

from models.vduq import vDUQParams
from models.model_params import GPParams, NNParams, OptimizerParams, TrainingParams
from experimental.experiment_params import ExperimentParams
from methods.random import RandomParams
from datasets.activelearningdataset import DatasetName
from utils.config import IO

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
    var_optimizer = None
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
    parser = init_parser()
    args = parser.parse_args()
    file_name = "experiments/" + args.name + "/model.json"
    if IO.file_exists(file_name):
        json_str = IO.load_dict_from_file(file_name)
        expr_config = IO.parseParams(ExperimentParams, json_str)
    else:
        expr_config = ExperimentParams(
                model_params =  vDUQParams(
                    training_params = training_params,
                    fe_params = nn_params,
                    gp_params = gp_params
                ),
                method_params = RandomParams(
                    batch_size = 0,
                    max_num_batches = 5,
                    initial_size = 60*bs
                ),
                dataset_params = DatasetParams(
                    batch_size = bs
                )
        )
        log = expr_config.export()
        IO.dict_to_file(log, file_name)

    expr = Experiment(
        args.name,
        "Testing Experimental Framework",
        expr_config
    )
    expr.run()