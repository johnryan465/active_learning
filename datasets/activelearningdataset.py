from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Sequence
import torch
from batchbald_redux.active_learning import RandomFixedLengthSampler
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from torchvision.datasets.vision import VisionDataset
from .dataset_params import DatasetParams
from torchtyping import TensorType # type: ignore


from typeguard import typechecked

Indexes = List[int]

class DatasetName(str, Enum):
    cifar10 = 'cifar10'
    mnist = 'mnist'


# Datasets which we want to be able to work with should implment this interface

class ActiveLearningDataset(ABC):
    @abstractmethod
    def get_train(self) -> DataLoader:
        pass

    @abstractmethod
    def get_test(self) -> DataLoader:
        pass

    @abstractmethod
    def get_pool(self) -> DataLoader:
        pass

    @abstractmethod
    def get_pool_tensor(self) -> torch.Tensor:
        pass

    @abstractmethod
    def move(self, idx: Indexes):
        pass

    @abstractmethod
    def get_classes(self) -> List[str]:
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

class DatasetWrapper(ActiveLearningDataset):
    num_workers = 4
    
    def __init__(self, train_dataset: VisionDataset,
                 test_dataset: VisionDataset, config: DatasetParams) -> None:
        super().__init__()
        self.bs = config.batch_size

        if config.smoke_test:
            self.trainset = Subset(train_dataset, list(range(0,100)))
            self.testset = Subset(test_dataset, list(range(0,100))) # test_dataset
            self.sampler_size = 4000
        else:
            self.trainset = train_dataset
            self.testset = test_dataset
            self.sampler_size = 40000
        if config.num_repetitions > 1:
            self.trainset = ConcatDataset([self.trainset] * config.num_repetitions)


        self.test_loader = DataLoader(
            self.testset, batch_size=self.bs, shuffle=False,
            pin_memory=True,
            drop_last=True, num_workers=DatasetWrapper.num_workers)

        self.classes = config.classes.split(',')
        self.pool_size = len(self.trainset)
        self.mask = torch.zeros(self.pool_size)

    def get_train(self) -> DataLoader:
        ts = Subset(self.trainset, DatasetUtils.tensor_to_sequence(torch.nonzero(self.mask != 0).squeeze()))
        return DataLoader(
            ts, batch_size=self.bs, num_workers=DatasetWrapper.num_workers, drop_last=False,
            pin_memory=True,
            sampler=RandomFixedLengthSampler(ts, self.sampler_size)
        )

    def get_test(self):
        return self.test_loader

    def get_pool(self) -> DataLoader:
        ts = Subset(self.trainset, DatasetUtils.tensor_to_sequence(torch.nonzero(self.mask == 0).squeeze()))
        return DataLoader(
            ts, batch_size=self.bs, num_workers=DatasetWrapper.num_workers, shuffle=True, pin_memory=True
        )

    def get_pool_tensor(self) -> Dataset:
        ts = Subset(
            self.trainset, DatasetUtils.tensor_to_sequence(torch.nonzero(self.mask == 0).squeeze()))
        return ts
    
    # This method and related ones should take inputs in the range
    # 0 - poolsize instead of 0 - dataset size
    def move(self, idxs: Indexes) -> None:
        pool_idxs = torch.nonzero(self.mask == 0)
        for i in idxs:
            self.mask[pool_idxs[i].unsqueeze(0)] = 1
        self.pool_size -= len(idxs)

    def get_classes(self) -> List[str]:
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
    
    @staticmethod
    def tensor_to_sequence(input: torch.Tensor) -> Sequence[int]:
        return input.tolist()
