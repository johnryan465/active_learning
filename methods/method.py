from abc import ABC, abstractmethod

from models.model import ModelWrapper
from datasets.activelearningdataset import ActiveLearningDataset


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
    pass
