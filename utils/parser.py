from uncertainty.estimator_entropy import Sampling
from models.bnn import BNNParams
from methods.BatchBALD import BatchBALDParams
from methods.BALD import BALDParams
from methods.Entropy import EntropyParams
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
            pass
        elif method == "batchbald":
            pass
        elif method == "bald":
            pass
        elif method == "random":
            pass
        p.add_argument('--batch_size', type=int, required=True)
        p.add_argument('--epochs', default=100, type=int, required=True)
        p.add_argument('--dataset', default=DatasetName.mnist, type=DatasetName)
        p.add_argument('--num_repetitions', default=1, type=int)
        p.add_argument('--unbalanced', default=False, type=bool)

        
        p.set_defaults(method=method)
        nestedsubpraser = p.add_subparsers(dest='model')
        for model in models:
            q = nestedsubpraser.add_parser(model)
            q.set_defaults(model=model)
            q.add_argument('--model_index', default=0, type=int)
            q.add_argument('--var_opt', default=-1, type=float)
            q.add_argument('--dropout', default=0.0, type=float)
            q.add_argument('--lr', default=0.01, type=float)
            q.add_argument('--spectral_norm', default=False, action='store_true')
            q.add_argument('--power_iter', default=1, type=int)
            q.add_argument('--coeff', default=9, type=float)
            q.add_argument('--n_inducing_points', default=10, type=int)
            if model == "vduq":
                q.add_argument('--ard', default=-1, type=int)
            elif model == "bnn":
                pass
            elif model == "dnn":
                pass
            else:
                pass
        method_parsers[method] = p

    return parser


"""
These functions take an Argparser namespace and parse it into the internal configs we use

This level of abstraction lets us keep the command line interface separate from the configs
"""


def parse_dataset(args: argparse.Namespace) -> DatasetParams:
    # Setup the dataset config
    if args.unbalanced:
        weights = tuple([1.0] + ([0.2] * 9))
    else:
        weights = tuple([])
    if args.dataset == DatasetName.mnist:            
        dataset_params = DatasetParams(
            path=args.data_path,
            batch_size=args.batch_size,
            num_repetitions=args.num_repetitions,
            smoke_test=args.smoke_test,
            class_weighting=weights
        )
    else:
        dataset_params = DatasetParams(
            path=args.data_path,
            batch_size=args.batch_size,
            num_repetitions=args.num_repetitions,
            smoke_test=args.smoke_test,
            class_weighting=weights
        )
    return dataset_params


def parse_method(args: argparse.Namespace) -> MethodParams:
    # Create the active learning method
    if args.model == ModelName.vduq:
        sampling_config = Sampling(batch_samples=400, per_samples=30, sum_samples=20)
    else:
        sampling_config = Sampling(batch_samples=30, per_samples=10, sum_samples=200)

    if args.method == MethodName.batchbald:
        method_params = BatchBALDParams(
            aquisition_size=args.aquisition_size,
            max_num_aquisitions=args.num_aquisitions,
            initial_size=args.initial_per_class,
            samples = sampling_config,
            use_cuda=use_cuda,
            smoke_test=args.smoke_test
        )
    elif args.method == MethodName.bald:
        method_params = BALDParams(
            aquisition_size=args.aquisition_size,
            max_num_aquisitions=args.num_aquisitions,
            initial_size=args.initial_per_class,
            samples=sampling_config,
            smoke_test=args.smoke_test
        )
    elif args.method == MethodName.entropy:
        method_params = EntropyParams(
            aquisition_size=args.aquisition_size,
            max_num_aquisitions=args.num_aquisitions,
            initial_size=args.initial_per_class,
            samples=sampling_config,
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
            spectral_normalization=args.spectral_norm,
            dropout_rate=args.dropout,
            coeff=args.coeff,
            n_power_iterations=1,
            batchnorm_momentum=0.01,
            weight_decay=5e-4,
        )

        model_params = vDUQParams(
            model_index=args.model_index,
            gp_params=gp_params,
            fe_params=nn_params
        )
    elif args.model == ModelName.bnn:
        nn_params = NNParams(
            dropout_rate=args.dropout,
            batchnorm_momentum=0.01,
            weight_decay=5e-4,
        )

        model_params = BNNParams(
            model_index=args.model_index,
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
            num_classes=10
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
