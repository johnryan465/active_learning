from types import FunctionType

from params.model_params import NNParams, TrainingParams
from datasets.activelearningdataset import ActiveLearningDataset
from .model import ModelWrapper
from params.model_params import ModelParams

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from marshmallow_dataclass import dataclass


class DNNParams(ModelParams):
    def __init__(self, training_params: TrainingParams,
                 nn_params: NNParams) -> None:
        super().__init__()
        self.training_params = training_params
        self.nn_params = nn_params

    def toDict(self) -> str:
        return self.__dict__


class DNN(ModelWrapper):
    def __init__(self, params: DNNParams) -> None:
        super().__init__()
        self.params = params
        if params.training_params.dataset == "MNIST":
            self.model = MNISTNet()
        elif params.training_params.dataset == "CIFAR10":
            self.model = CIFAR10Net()
        else:
            raise ValueError('Unsupported Dataset')

        if self.params.training_params.cuda:
            self.model = self.model.cuda()
        
        self.paramaters = [{"params":self.model.parameters(), "lr":0.001}]
        self.optimizer = optim.SGD(self.paramaters, momentum=0.9)
        self.loss_fn = nn.CrossEntropyLoss()


    def reset(self) -> None:
        self.net = CIFAR10Net()
    
    def get_loss_fn(self):
        return self.loss_fn

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer
    
    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_model_params(self) -> dict:
        return self.params
    
    def prepare(self, batch_size: int) -> None:
        pass

    def get_training_params(self) ->  TrainingParams:
        return self.params.training_params

    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return None

    def get_eval_step(self) -> FunctionType:
        def eval_step(engine, batch):
            self.model.eval()

            x, y = batch
            if self.params.training_params.cuda:
                x, y = x.cuda(), y.cuda()

            with torch.no_grad():
                y_pred = self.model(x)

            return y_pred, y
        return eval_step

    def get_train_step(self):
        optimizer = self.optimizer
        def step(engine, batch):
            self.model.train()

            optimizer.zero_grad()
            x, y = batch
            if self.params.training_params.cuda:
                x, y = x.cuda(), y.cuda()

            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            return loss.item()
        return step
    
    def get_output_transform(self):
        return lambda x : x


class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
