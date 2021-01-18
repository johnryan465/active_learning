from abc import ABC, abstractmethod
from enum import Enum
from typing import List
import torch
from torch.utils.data import sampler
import numpy as np
Indexes = List[int]

class DatasetName(str,Enum):
    cifar10='cifar10'
    mnist='mnist'


# Datasets which we want to be able to work with should implment this interface

class ActiveLearningDataset(ABC):
    @abstractmethod
    def get_train(self):
        pass

    @abstractmethod
    def get_test(self):
        pass

    @abstractmethod
    def get_pool(self):
        pass

    @abstractmethod
    def move(self, idx: Indexes):
        pass

    @abstractmethod
    def get_classes(self):
        pass

    @abstractmethod
    def get_name(self) -> DatasetName:
        pass

    @abstractmethod
    def get_pool_size(self) -> int:
        pass




# This is a simple wrapper which can be used to make pytorch datasets easily correspond to the interface above

class DatasetWrapper(ABC):
    def __init__(self, train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset, batch_size: int) -> None:
        super().__init__()
        self.bs = batch_size
        self.trainset = train_dataset
        self.testset = test_dataset

        self.test_loader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.bs, shuffle=False,
            drop_last=True, num_workers=1)

        self.classes = train_dataset.classes
        self.pool_size = len(self.trainset)
        self.mask = torch.zeros(self.pool_size)

    def get_train(self):
        return torch.utils.data.DataLoader(
            self.trainset, batch_size=self.bs, num_workers=2 ,drop_last=False,
            sampler=sampler.SubsetRandomSampler(
                torch.nonzero(self.mask).squeeze())
        )

    def get_test(self):
        return self.test_loader

    def get_pool(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.trainset, batch_size=self.bs, num_workers=2,
            sampler=sampler.SubsetRandomSampler(
                torch.nonzero(self.mask == 0).squeeze()
            )
        )

    # This method and related ones should take inputs in the range
    # 0 - poolsize instead of 0 - dataset size
    def move(self, idxs: Indexes) -> None:
        pool_idxs = torch.nonzero(self.mask == 0)
        for i in idxs:
            self.mask[pool_idxs[i]] = 1
        self.pool_size -= len(idxs)

    def get_classes(self):
        return self.classes
    
    def get_pool_size(self):
        return self.pool_size
