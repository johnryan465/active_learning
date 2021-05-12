from experimental.experiment import Experiment

from experimental.experiment_params import ExperimentParams
from utils.config import IO

from utils.parser import parse_dataset, parse_model, parse_training, parse_method, init_parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    print(args)
    file_name = "experiments/" + args.name + "/model.json"
    if IO.file_exists(file_name):
        json_str = IO.load_dict_from_file(file_name)
        expr_config = IO.parseParams(ExperimentParams, json_str)
    else:
        # Create the active learning method
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

        log = expr_config.export()
        # IO.dict_to_file(log, file_name)

    expr = Experiment(
        args.name,
        args.description,
        expr_config
    )
    expr.run()
