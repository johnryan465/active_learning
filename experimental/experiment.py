from methods.BatchBALD import BatchBALD, BatchBALDParams
from methods.BALD import BALD, BALDParams
from methods.Entropy import Entropy, EntropyParams
from models.bnn import BNN, BNNParams
from methods.random import Random, RandomParams
from datasets.mnist import MNIST
from models.dnn import DNN, DNNParams
from models.vduq import vDUQ, vDUQParams
from models.model import ModelWrapper
from methods.method import Method
from datasets.activelearningdataset import ActiveLearningDataset
from experimental.experiment_params import ExperimentParams
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from models.model_params import ModelParams
from models.training import TrainingParams
from methods.method_params import MethodParams
from datasets.dataset_params import DatasetParams
from .driver import Driver
import time




# This is responsible for actually creating and executing an experiment

class Experiment:
    def __init__(self, name: str, objective: str, experiment_params: ExperimentParams):
        self.bs = experiment_params.dataset_params.batch_size

        self.dataset = Experiment.create_dataset(experiment_params.dataset_params)
        self.method = Experiment.create_method(experiment_params.method_params)
        self.method.initialise(self.dataset)
        self.model = Experiment.create_model(
            experiment_params.model_params,
            experiment_params.training_params, self.dataset)
        self.name = name
        self.objective = objective
        self.training_params = experiment_params.training_params

    @staticmethod
    def create_model(model_config: ModelParams, training_config: TrainingParams, dataset: ActiveLearningDataset) -> ModelWrapper:
        if isinstance(model_config, vDUQParams):
            model = vDUQ(model_config, training_config, dataset)
        elif isinstance(model_config, DNNParams):
            model = DNN(model_config, training_config, dataset)
        elif isinstance(model_config, BNNParams):
            model = BNN(model_config, training_config, dataset)
        else:
            raise NotImplementedError('Model')
        return model

    @staticmethod
    def create_method(method_config: MethodParams) -> Method:
        if isinstance(method_config, RandomParams):
            method = Random(method_config)
        elif isinstance(method_config, BALDParams):
            method = BALD(method_config)
        elif isinstance(method_config, BatchBALDParams):
            method = BatchBALD(method_config)
        elif isinstance(method_config, EntropyParams):
            method = Entropy(method_config)
        else:
            raise NotImplementedError('Method')
        return method

    @staticmethod
    def create_dataset(dataset_config: DatasetParams) -> ActiveLearningDataset:
        dataset = MNIST(dataset_config)
        return dataset

    def run(self) -> None:
        iteration = 0
        while(not self.method.complete()):
            ts = time.time()
            name = self.name + "_" + str(iteration) + str(ts)
            tb_logger = TensorboardLogger(flush_secs=1, log_dir="logs/" + name)
            self.model = Driver.train(name, iteration, self.training_params, self.model, self.dataset, tb_logger)
            self.method.acquire(self.model, self.dataset, tb_logger)
            self.model.prepare(self.bs)
            self.model.initialize(self.dataset)
            tb_logger.close()
            iteration = iteration + 1
