from methods.BatchBALD import BatchBALD, BatchBALDParams
from methods.BALD import BALD, BALDParams
from models.bnn import BNN, BNNParams
from methods.random import Random, RandomParams
from methods.BALD import BALD, BALDParams
from datasets.mnist import MNIST
from models.dnn import DNN, DNNParams
from models.vduq import vDUQ, vDUQParams
from models.model import ModelWrapper
from methods.method import Method
from datasets.activelearningdataset import ActiveLearningDataset
from experimental.experiment_params import ExperimentParams
from models.model_params import ModelParams
from methods.method_params import MethodParams
from datasets.dataset_params import DatasetParams
from .driver import Driver


# This is responsible for actually creating and executing an experiment

class Experiment:
    def __init__(self, name : str, objective : str, experiment_params : ExperimentParams):
        self.bs = experiment_params.dataset_params.batch_size

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
            model = DNN(model_config, dataset)
        elif isinstance(model_config, BNNParams):
            model = BNN(model_config, dataset)
        else:
            model = DNN(model_config, dataset)
        return model

    @staticmethod
    def create_method(method_config : MethodParams) -> Method:
        if isinstance(method_config, RandomParams):
            method = Random(method_config)
        elif isinstance(method_config, BALDParams):
            method = BALD(method_config)
        elif isinstance(method_config, BatchBALDParams):
            method = BatchBALD(method_config)
        else:
            method = Random(method_config)
        return method

    @staticmethod
    def create_dataset(dataset_config : DatasetParams) -> ActiveLearningDataset:
        dataset = MNIST(dataset_config.batch_size)
        return dataset

    def run(self) -> None:
        iteration = 0
        while(not self.method.complete()):
            self.model.reset(self.dataset)
            Driver.train(self.name, iteration, self.model, self.dataset)
            self.method.acquire(self.model, self.dataset)
            self.model.prepare(self.bs)
            iteration = iteration + 1

