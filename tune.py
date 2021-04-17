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
        lr = config["point"][0]
        dropout = config["dropout"]
        method = config["method"]
        coeff = config["point"][1]
        batch_size = config["batch_size"]
        var_opt = config["var_opt"]
        starting_size = config["starting_size"]
        num_aquisitions = config["num_aquisitions"]


        # aquisition
        args = Namespace(
            data_path=path,
            aquisition_size=4, batch_size=batch_size, dataset=DatasetName.mnist, description='ray-vduq', dropout=dropout,
            epochs=500, initial_per_class=starting_size, smoke_test=False, lr=lr, method=method, use_progress=False, model='vduq', model_index=0, var_opt=var_opt, num_repetitions=4, name='vduq_bb_tuning',
            num_aquisitions=num_aquisitions, power_iter=1, spectral_norm=True, coeff=coeff)

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

    df_search = DragonflySearch(
        optimizer="bandit",
        domain="euclidean")

    df_search = ConcurrencyLimiter(df_search, max_concurrent=4)
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=70,
        brackets=1,
        grace_period=10,
        reduction_factor=3
    )

    analysis = tune.run(
        create_training_function(args.data_path),
        metric="accuracy",
        mode="max",
        search_alg=df_search,
        scheduler=scheduler,
        num_samples=10,
        resources_per_trial={'gpu': 1},
        config={
            "lr": tune.loguniform(5e-4, 1e-1),
            "dropout": 0.0,
            "method": "batchbald",
            "coeff": tune.uniform(6, 12),
            "batch_size": 64,
            "starting_size": 2,
            "num_aquisitions": 70,
            "var_opt": -1
        })
    print(analysis)
    print("Best config: ", analysis.get_best_config(
        metric="mean_loss", mode="min"))
