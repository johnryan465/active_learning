from abc import ABC, abstractmethod
from enum import Enum
from typing import List
import torch
from batchbald_redux.active_learning import RandomFixedLengthSampler
from .dataset_params import DatasetParams
Indexes = List[int]


class DatasetName(str, Enum):
    cifar10 = 'cifar10'
    mnist = 'mnist'


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

    @abstractmethod
    def get_indexes(self, idxs: Indexes) -> List[torch.Tensor]:
        pass


# This is a simple wrapper which can be used to make pytorch datasets easily correspond to the interface above

class DatasetWrapper(ABC):
    num_workers = 4
    def __init__(self, train_dataset: torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset, config: DatasetParams) -> None:
        super().__init__()
        self.bs = config.batch_size
        self.trainset = train_dataset

        if config.num_repetitions > 1:
            self.trainset = torch.utils.data.ConcatDataset([train_dataset] * config.num_repetitions)
        else:
            self.trainset = train_dataset

        self.testset = test_dataset

        self.test_loader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.bs, shuffle=False,
            pin_memory=True,
            drop_last=True, num_workers=DatasetWrapper.num_workers)

        self.classes = train_dataset.classes
        self.pool_size = len(self.trainset)
        self.mask = torch.zeros(self.pool_size)

    def get_train(self):
        ts = torch.utils.data.Subset(
            self.trainset, torch.nonzero(self.mask != 0).squeeze())
        return torch.utils.data.DataLoader(
            ts, batch_size=self.bs, num_workers=DatasetWrapper.num_workers, drop_last=False,
            pin_memory=True,
            sampler=RandomFixedLengthSampler(ts, 40000)
        )

    def get_test(self):
        return self.test_loader

    def get_pool(self) -> torch.utils.data.DataLoader:
        ts = torch.utils.data.Subset(
            self.trainset, torch.nonzero(self.mask == 0).squeeze())
        return torch.utils.data.DataLoader(
            ts, batch_size=self.bs, num_workers=DatasetWrapper.num_workers, shuffle=True, pin_memory=True
        )

    # This method and related ones should take inputs in the range
    # 0 - poolsize instead of 0 - dataset size
    def move(self, idxs: Indexes) -> None:
        pool_idxs = torch.nonzero(self.mask == 0)
        for i in idxs:
            self.mask[pool_idxs[i].unsqueeze(0)] = 1
        self.pool_size -= len(idxs)

    def get_classes(self):
        return self.classes

    def get_pool_size(self):
        return self.pool_size

    def get_indexes(self, idxs: Indexes) -> List[torch.Tensor]:
        pool_idxs = torch.nonzero(self.mask == 0)
        res = []
        for i in idxs:
            image, label = self.trainset[pool_idxs[i].squeeze()]
            res.append(image)
        return res

class DatasetUtils:
    @staticmethod
    def balanced_init(dataset: ActiveLearningDataset, per_class: int):
        num_classes = len(dataset.get_classes())
        current_sample_idx = 0
        collected_indexes = {}
        indexes = []
        completed_classes = 0
        for i in range(0, num_classes):
            collected_indexes[i] = 0

        for _, y in dataset.get_pool():
            for i in range(len(y)):
                yi = y[i].item()
                if collected_indexes[yi] < per_class:
                    collected_indexes[yi] += 1
                    if collected_indexes[yi] == per_class:
                        completed_classes += 1

                    indexes.append(current_sample_idx)

                current_sample_idx += 1
            if completed_classes == num_classes:
                break
        dataset.move(indexes)
