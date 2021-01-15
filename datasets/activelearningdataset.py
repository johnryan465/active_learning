from abc import ABC, abstractmethod
from typing import List
import torch
from torch.utils.data import sampler

Indexes = List[int]


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


class DatasetWrapper(ABC):
    def __init__(self, train_dataset, test_dataset, batch_size: int) -> None:
        super().__init__()
        self.bs = batch_size

        self.trainset = torch.utils.data.Subset(
            train_dataset, list(range(0, 40000)))
        # self.trainset = train_dataset

        self.testset = torch.utils.data.Subset(
            test_dataset, list(range(0, 10000)))
        # self.testset = test_dataset

        self.test_loader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.bs, shuffle=False,
            drop_last=True, num_workers=1)

        self.classes = (
            'plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.mask = torch.zeros((len(self.trainset)))

    def get_train(self):
        return torch.utils.data.DataLoader(
            self.trainset, batch_size=self.bs, num_workers=1 ,drop_last=True,
            sampler=sampler.SubsetRandomSampler(
                torch.nonzero(self.mask))
        )

    def get_test(self):
        return self.test_loader

    def get_pool(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.trainset, batch_size=self.bs, num_workers=1,
            sampler=sampler.SubsetRandomSampler(
                torch.nonzero(self.mask == 0)
            )
        )

    def move(self, idxs: Indexes) -> None:
        for i in idxs:
            self.mask[i] = 1

    def get_classes(self):
        return self.classes
