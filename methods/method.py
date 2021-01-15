from abc import ABC, abstractmethod

from models.model import ModelWrapper
from datasets.activelearningdataset import ActiveLearningDataset


class Method(ABC):
    @abstractmethod
    def acquire(self, model: ModelWrapper, dataset: ActiveLearningDataset) -> None:
        """
        returns a pytorch training dataset
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
        returns a pytorch training dataset
        """
        pass


class UncertainMethod(Method):
    pass
