from models.bnn import BNNParams
from methods.random import Random
from datasets.mnist import MNIST
from models.dnn import DNN, DNNParams
from models.vduq import vDUQ, vDUQParams
from models.model import ModelWrapper
from methods.method import Method
from datasets.activelearningdataset import ActiveLearningDataset
from params.experiment_params import ExperimentParams
from params.model_params import GPParams, ModelParams, OptimizerParams, TrainingParams, NNParams
from params.method_params import MethodParams
from params.dataset_params import DatasetParams
from models.driver import Driver


# This is responsible for actually creating and executing an experiment

class Experiment:
    def __init__(self, name : str, objective : str, experiment_params : ExperimentParams):
        self.bs = 256
        self.dataset = Experiment.create_dataset(experiment_params.dataset_params)
        self.method = Experiment.create_method(experiment_params.method_params)
        self.method.initialise(self.dataset)
        self.model = Experiment.create_model(experiment_params.model_params,self.dataset)
        self.name = name
        self.objective = objective

    @staticmethod
    def create_model(model_config : ModelParams, dataset : ActiveLearningDataset) -> ModelWrapper:
        if isinstance(model_config, vDUQParams):
            model = vDUQ(model_config, dataset)
        elif isinstance(model_config, DNNParams):
            model = DNN(model_config)
        elif isinstance(model_config, BNNParams):
            model = BNN(model_config)
        return model

    @staticmethod
    def create_method(method_config : MethodParams) -> Method:
        method = Random(method_config.batch_size, 1,method_config.batch_size*60)
        return method

    @staticmethod
    def create_dataset(dataset_config : DatasetParams) -> ActiveLearningDataset:
        dataset = MNIST(dataset_config.batch_size)
        return dataset

    def run(self) -> None:
        iteration = 0
        while(not self.method.complete()):
            # model.reset()
            Driver.train(self.name, iteration, self.model, self.dataset)
            # Driver.test(model, dataset)
            self.method.acquire(self.model, self.dataset)
            self.model.prepare(self.bs)
            iteration = iteration + 1
            break

