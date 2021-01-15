from abc import ABC, abstractmethod
from types import FunctionType

import torch


class ModelWrapper(ABC):
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the model to an untrained state
        """
        pass

    @abstractmethod
    def prepare(self, batch_size: int) -> None:
        """
        Prepares the model for the next round
        """
        pass

    @abstractmethod
    def get_model_params(self) -> dict:
        """
        Returns the trainable model parameters
        """
        pass

    @abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        """
        Returns the optimizer we are using in training
        """
        pass

    @abstractmethod
    def get_train_step(self) -> FunctionType:
        pass

    @abstractmethod
    def get_eval_step(self) -> FunctionType:
        pass

    @abstractmethod
    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        pass

    @abstractmethod
    def get_output_transform(self):
        pass

    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        """
        Returns the pytorch model which the class wraps around
        """
        pass

    @abstractmethod
    def get_training_params(self):
        pass

    @abstractmethod
    def get_loss_fn(self):
        pass

# This is a model which we can sample from
class UncertainModel(ModelWrapper):
    @abstractmethod
    def sample(self, input: torch.tensor, samples: int):
        """
        returns a pytorch test dataset
        """
        pass