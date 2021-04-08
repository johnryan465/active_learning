from experimental.experiment import Experiment
from experimental.experiment_params import ExperimentParams
from datasets.activelearningdataset import DatasetName
from utils.parser import parse_dataset, parse_model, parse_training, parse_method

import ray
from ray import tune
from argparse import Namespace
import argparse
import uuid


def create_training_function(path):
    def training_function(config):

        # Hyperparameters
        lr = config["lr"]
        dropout = config["dropout"]
        method = config["method"]
        coeff = config["coeff"]
        batch_size = config["batch_size"]
        var_opt = config["var_opt"]


        # aquisition
        args = Namespace(
            data_path=path,
            aquisition_size=4, batch_size=batch_size, dataset=DatasetName.mnist, description='ray-vduq', dropout=dropout,
            epochs=500, initial_per_class=5000, lr=lr, method=method, use_progress=False, model='vduq', model_index=0, var_opt=var_opt, num_repetitions=1, name='vduq_bb_tuning',
            num_aquisitions=1, power_iter=1, spectral_norm=True, coeff=coeff)

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
    parser.add_argument('--data_path', default="./data", type=str)
    args = parser.parse_args()
    ray.init(include_dashboard=True)
    analysis = tune.run(
        create_training_function(args.data_path),
        resources_per_trial={'gpu': 1},
        num_samples=4,
        config={
            "lr": tune.grid_search([0.03]),
            "dropout": tune.grid_search([0.0]),
            "method": tune.choice(["batchbald"]),
            "coeff": tune.grid_search([9]),
            "batch_size":  tune.grid_search([64]),
            "var_opt": tune.choice([None]),
        })
    print(analysis)
    print("Best config: ", analysis.get_best_config(
        metric="mean_loss", mode="min"))
