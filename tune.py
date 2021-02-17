from experimental.experiment import Experiment
from experimental.experiment_params import ExperimentParams
from datasets.activelearningdataset import DatasetName
from utils.parser import parse_dataset, parse_model, parse_training, parse_method

import ray
from ray import tune
from argparse import Namespace
import uuid


def training_function(config):
    # Hyperparameters
    lr = config["lr"]
    dropout = config["dropout"]
    method = config["method"]
    coeff = config["coeff"]
    args = Namespace(
        aquisition_size=5, batch_size=64, dataset=DatasetName.mnist, description='ray-vduq', dropout=dropout,
        epochs=500, initial_per_class=2, lr=lr, method=method, use_progress=False, model='vduq', model_index=0, name='vduq_bb_tuning',
        num_aquisitions=10, power_iter=1, spectral_norm=True, coeff=coeff)

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


if __name__ == "__main__":
    ray.init(include_dashboard=False)
    analysis = tune.run(
        training_function,
        resources_per_trial={'gpu': 1},
        num_samples=10,
        config={
            "lr": tune.grid_search([0.001, 0.01, 0.1]),
            "dropout": tune.grid_search([0.1, 0.3, 0.5, 0.7]),
            "method": tune.choice(["random", "batchbald", "bald"]),
            "coeff": tune.grid_search([1.5, 3, 9]),
        })
    print(analysis)
    print("Best config: ", analysis.get_best_config(
        metric="mean_loss", mode="min"))
