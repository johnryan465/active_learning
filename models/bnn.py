from types import FunctionType
from typing import Callable, Dict
from models.model_params import NNParams, TrainingParams, ModelParams
from datasets.activelearningdataset import ActiveLearningDataset, DatasetName
from models.model import UncertainModel
from uncertainty.fixed_dropout import BayesianModule, ConsistentMCDropout2d
from uncertainty.fixed_dropout import ConsistentMCDropout

import torch
from torch import nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


class BayesianMNIST(BayesianModule):
    def __init__(self, params: NNParams):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = ConsistentMCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = ConsistentMCDropout()
        self.fc2 = nn.Linear(128, 10)

    def mc_forward_impl(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)
        return input


@dataclass
class BNNParams(ModelParams):
    training_params: TrainingParams
    fe_params: NNParams


class BNN(UncertainModel):
    model_config = {
        DatasetName.mnist: [BayesianMNIST],
        DatasetName.cifar10: []
    }

    def __init__(self, params: BNNParams, dataset: ActiveLearningDataset) -> None:
        super().__init__()
        self.nn_params = params.fe_params
        self.training_params = params.training_params
        self.reset(dataset)

    def get_eval_step(self) -> FunctionType:
        def eval_step(engine, batch):
            self.model.eval()

            x, y = batch
            if self.training_params.cuda:
                x, y = x.cuda(), y.cuda()

            with torch.no_grad():
                y_pred = self.model(x, 1)

            return y_pred.squeeze(), y
        return eval_step

    def get_train_step(self) -> FunctionType:
        def step(engine, batch):
            self.model.train()

            self.optimizer.zero_grad()
            x, y = batch
            if self.training_params.cuda:
                x, y = x.cuda(), y.cuda()
            y_pred = self.model(x, 1)
            y_pred = y_pred.squeeze()
            loss = self.get_loss_fn()(y_pred, y)
            loss.backward()
            self.optimizer.step()

            return loss.item()
        return step

    def get_loss_fn(self):
        def loss_fn(x, y):
            s = nn.NLLLoss()
            x = x.squeeze()
            return s(x, y)
        return loss_fn

    def sample(self, input: torch.tensor, samples: int):
        return self.model(input, samples)

    def reset(self, dataset: ActiveLearningDataset) -> None:
        self.model = BNN.model_config[self.training_params.dataset][self.training_params.model_index](self.nn_params)

        self.parameters = [
            {"params": self.model.parameters(),
             "lr": self.training_params.optimizers.optimizer}]

        milestones = [60, 120, 160]
        self.optimizer = torch.optim.SGD(
            self.parameters, momentum=0.9, weight_decay=self.nn_params.weight_decay
        )
        if self.training_params.cuda:
            self.model = self.model.cuda()

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.2
        )

    def get_output_transform(self):
        return lambda x: x

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_model_params(self) -> dict:
        return self.parameters

    def get_training_params(self):
        return self.training_params

    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return self.scheduler

    def get_training_log_hooks(self) -> Dict[str, Callable[[Dict[str, float]], float]]:
        return {
            'loss': lambda x: x
        }

    def get_test_log_hooks(self) -> Dict[str, Callable[[Dict[str, float]], float]]:
        return {
            'accuracy': self.get_output_transform(),
            'loss': lambda x: x
        }

    def prepare(self, batch_size: int) -> None:
        return super().prepare(batch_size)
