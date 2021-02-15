from abc import ABC, abstractmethod
from enum import Enum

from models.model import ModelWrapper, UncertainModel
from datasets.activelearningdataset import ActiveLearningDataset


class MethodName(str, Enum):
    batchbald = 'batchbald'
    bald = 'bald'
    random = 'random'


class Method(ABC):
    @abstractmethod
    def acquire(self, model: ModelWrapper, dataset: ActiveLearningDataset) -> None:
        """
        Moves data from the pool to training
        """
        pass

    @abstractmethod
    def complete(self) -> bool:
        """
        Checks whether or not the method is complete
        """
        pass

    @abstractmethod
    def initialise(self, dataset: ActiveLearningDataset) -> None:
        """
        Initialise the dataset to have some training data
        """
        pass


# A type of method which requires a model which can output
# with uncertainty
class UncertainMethod(Method):
    @abstractmethod
    def acquire(self, model: UncertainModel, dataset: ActiveLearningDataset) -> None:
        pass
