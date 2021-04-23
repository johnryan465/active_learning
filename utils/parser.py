from methods.BatchBALD import BatchBALDParams
from methods.BALD import BALDParams
from methods.random import RandomParams

from methods.method import MethodName
from methods.method_params import MethodParams
from datasets.dataset_params import DatasetParams

from models.vduq import vDUQParams
from models.dnn import DNNParams
from models.model_params import GPParams, NNParams, ModelParams
from models.model import ModelName
from models.training import TrainingParams, OptimizerParams
from datasets.activelearningdataset import DatasetName

import argparse
import torch

use_cuda = torch.cuda.is_available()


def init_parser() -> argparse.ArgumentParser:
    """
    Each method and model each have paramaters which should be inputed via cmd
    But only certain methods and models are compatible with each other

    We take the methods as the highest level of description
    """
    parser = argparse.ArgumentParser(description="Run experiments for active learning")
    subprasers = parser.add_subparsers(dest='method')
    parser.add_argument('--name', required=True)
    parser.add_argument('--description', required=True)
    parser.add_argument('--aquisition_size', default=4, type=int, required=True)
    parser.add_argument('--num_aquisitions', default=10, type=int)
    parser.add_argument('--initial_per_class', default=2, type=int)
    parser.add_argument('--use_progress', default=True, type=bool)
    parser.add_argument('--smoke_test', default=False, type=bool)
    parser.add_argument('--data_path', default="./data", type=str)

    methods = ["entropy", "batchbald", "bald", "random"]
    models = ["vduq", "bnn", "dnn"]
    method_parsers = {}
    for method in methods:
        p = subprasers.add_parser(method)
        if method == "entropy":
            p.add_argument('--var_reduction', default=False, type=bool)
        elif method == "batchbald":
            p.add_argument('--var_reduction', default=False, type=bool)
        elif method == "bald":
            p.add_argument('--var_reduction', default=False, type=bool)
        elif method == "random":
            pass
        p.add_argument('--batch_size', type=int, required=True)
        p.add_argument('--epochs', default=100, type=int, required=True)
        p.add_argument('--dataset', default=DatasetName.mnist, type=DatasetName)
        p.add_argument('--num_repetitions', default=1, type=int)
        p.add_argument('--model_index', default=0, type=int)
        
        p.set_defaults(method=method)
        nestedsubpraser = p.add_subparsers(dest='model')
        for model in models:
            q = nestedsubpraser.add_parser(model)
            q.set_defaults(model=model)
            if model == "vduq":
                q.add_argument('--spectral_norm', default=True, type=bool)
                q.add_argument('--power_iter', default=1, type=int)
                q.add_argument('--dropout', default=0.0, type=float)
                q.add_argument('--lr', default=0.01, type=float)
                q.add_argument('--var_opt', default=-1, type=float)
                q.add_argument('--coeff', default=9, type=float)
                q.add_argument('--n_inducing_points', default=10, type=int)
            elif model == "bnn":
                q.add_argument('--dropout', default=0.0, type=float)
                q.add_argument('--lr', default=0.01, type=float)
            elif model == "dnn":
                q.add_argument('--dropout', default=0.0, type=float)
                q.add_argument('--lr', default=0.01, type=float)
            else:
                pass
        method_parsers[method] = p

    return parser


"""
These functions take an Argparser namespace and parse it into the internal configs we use

This level of abstraction lets us keep the command line interface seperate from the configs
"""


def parse_dataset(args: argparse.Namespace) -> DatasetParams:
    # Setup the dataset config
    if args.dataset == DatasetName.mnist:
        dataset_params = DatasetParams(
            path=args.data_path,
            batch_size=args.batch_size,
            num_repetitions=args.num_repetitions,
            smoke_test=args.smoke_test
        )
    else:
        dataset_params = DatasetParams(
            path=args.data_path,
            batch_size=args.batch_size,
            num_repetitions=args.num_repetitions,
            smoke_test=args.smoke_test
        )
    return dataset_params


def parse_method(args: argparse.Namespace) -> MethodParams:
    # Create the active learning method
    if args.method == MethodName.batchbald:
        method_params = BatchBALDParams(
            aquisition_size=args.aquisition_size,
            max_num_aquisitions=args.num_aquisitions,
            initial_size=args.initial_per_class,
            samples=100,
            use_cuda=use_cuda,
            var_reduction=args.var_reduction,
            smoke_test=args.smoke_test
        )
    elif args.method == MethodName.bald:
        method_params = BALDParams(
            aquisition_size=args.aquisition_size,
            max_num_aquisitions=args.num_aquisitions,
            initial_size=args.initial_per_class,
            samples=1,
            smoke_test=args.smoke_test
        )
    else:
        method_params = RandomParams(
            aquisition_size=args.aquisition_size,
            max_num_aquisitions=args.num_aquisitions,
            initial_size=args.initial_per_class,
            smoke_test=args.smoke_test
        )
    return method_params


def parse_model(args: argparse.Namespace) -> ModelParams:
    # print(args)
    if args.model == ModelName.vduq:
        gp_params = GPParams(
            kernel='RBF',
            num_classes=10,
            ard=-1,
            n_inducing_points=args.n_inducing_points,
            lengthscale_prior=False,
            separate_inducing_points=False,
            distribution="choskey"
        )

        nn_params = NNParams(
            spectral_normalization=True,
            dropout_rate=args.dropout,
            coeff=args.coeff,
            n_power_iterations=1,
            batchnorm_momentum=0.01,
            weight_decay=5e-4,
        )

        model_params = vDUQParams(
            model_index=0,
            gp_params=gp_params,
            fe_params=nn_params
        )
    else:
        nn_params = NNParams(
            spectral_normalization=args.spectral_norm,
            dropout_rate=args.dropout,
            coeff=args.coeff,
            n_power_iterations=args.power_iter,
            batchnorm_momentum=0.01,
            weight_decay=5e-4,
        )

        model_params = DNNParams(
            model_index=args.model_index,
            nn_params=nn_params
        )
    return model_params


def parse_training(args: argparse.Namespace) -> TrainingParams:
    # Parse training params
    opt_params = OptimizerParams(
        optimizer=args.lr,
        var_optimizer=args.var_opt
    )
    training_params = TrainingParams(
        dataset=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        cuda=use_cuda,
        optimizers=opt_params,
        patience=3,
        progress_bar=args.use_progress
    )
    return training_params
