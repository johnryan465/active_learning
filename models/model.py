from abc import ABC, abstractmethod
from models.training import TrainingParams
from datasets.activelearningdataset import ActiveLearningDataset
from types import FunctionType
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from enum import Enum
import torch

T = TypeVar("T")

class ModelWrapper(ABC):
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the model to an untrained state
        """
        pass

    @abstractmethod
    def initialize(self, dataset: ActiveLearningDataset) -> None:
        """
        Initialise the weights
        """
        pass

    @abstractmethod
    def prepare(self, batch_size: int) -> None:
        """
        Prepares the model for the next round
        """
        pass

    @abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        """
        Returns the optimizer we are using in training
        """
        pass

    @abstractmethod
    def get_train_step(self) -> Callable:
        pass

    @abstractmethod
    def get_eval_step(self) -> Callable:
        pass

    @abstractmethod
    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        pass

    @abstractmethod
    def get_output_transform(self) -> Callable:
        pass

    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        """
        Returns the pytorch model which the class wraps around
        """
        pass

    @abstractmethod
    def get_training_params(self) -> TrainingParams:
        pass

    @abstractmethod
    def get_loss_fn(self) -> Callable:
        pass

    @abstractmethod
    def get_training_log_hooks(self) -> Dict[str, Callable[[Dict[str, float]], float]]:
        pass

    @abstractmethod
    def get_test_log_hooks(self) -> Dict[str, Callable[[Dict[str, float]], float]]:
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def load_state_dict(cls: Type[T], state: Dict[str, Any], dataset: ActiveLearningDataset) -> T:
        pass

# This is a model which we can sample from
class UncertainModel(ModelWrapper):
    @abstractmethod
    def sample(self, input: torch.Tensor, samples: int) -> torch.Tensor:
        """
        returns tensor with the number of samples requested from the model
        """
        pass


class ModelName(str, Enum):
    bnn = 'bnn'
    vduq = 'vduq'
    dnn = 'dnn'
