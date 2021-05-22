from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.suggest.suggestion import ConcurrencyLimiter
from experimental.experiment import Experiment
from experimental.experiment_params import ExperimentParams
from datasets.activelearningdataset import DatasetName
from utils.parser import parse_dataset, parse_model, parse_training, parse_method
import ray
from ray import tune
from ray.tune.suggest.dragonfly import DragonflySearch
from argparse import Namespace
import argparse
import uuid


def create_training_function(path):
    def training_function(config):
        # Hyperparameters
        lr = config["lr"]
        dropout = config["dropout"]
        method = config["method"]
        model = config["model"]
        coeff = config["coeff"]
        batch_size = config["batch_size"]
        var_opt = config["var_opt"]
        starting_size = config["starting_size"]
        num_aquisitions = config["num_aquisitions"]
        n_inducing_points = config["n_inducing_points"]

        # aquisition
        args = Namespace(
            data_path=path,
            aquisition_size=10, batch_size=batch_size, dataset=DatasetName.mnist, description='ray-vduq', dropout=dropout,
            epochs=500, initial_per_class=starting_size, smoke_test=False, var_reduction=False, lr=lr, method=method, use_progress=False, model=model, model_index=0, var_opt=var_opt, n_inducing_points=n_inducing_points,
            num_repetitions=1, name='vduq_bb_tuning', num_aquisitions=num_aquisitions, power_iter=1, spectral_norm=True, coeff=coeff, unbalanced=True)

        dataset_params = parse_dataset(args)
        method_params = parse_method(args)
        model_params = parse_model(args)
        training_params = parse_training(args)

        expr_config = ExperimentParams(
                training_params=training_params,
                model_params=model_params,
                method_params=method_params,
                dataset_params=dataset_params
        )

        expr = Experiment(
            args.name + str(uuid.uuid4()),
            args.description,
            expr_config
        )
        expr.run()
    return training_function

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments for active learning")
    parser.add_argument('--data_path', default="/tmp/data", type=str)
    args = parser.parse_args()
    ray.init(include_dashboard=False)


    analysis = tune.run(
        create_training_function(args.data_path),
        metric="accuracy",
        mode="max",
        resources_per_trial={'gpu': 1},
        config={
            "lr": 0.003,
            "dropout": 0.0,
            "method": tune.grid_search(["batchbald","entropy","random","bald"]),
            "model": tune.grid_search(["vduq"]),
            "coeff": 9,
            "batch_size": 64,
            "starting_size": 2,
            "num_aquisitions": 30,
            "var_opt": 0.1,
            "n_inducing_points": 20,
        })
    print(analysis)
    print("Best config: ", analysis.get_best_config(
        metric="mean_loss", mode="min"))
