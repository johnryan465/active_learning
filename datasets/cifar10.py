import torchvision
import torchvision.transforms as transforms
from .activelearningdataset import DatasetWrapper


class CIFAR10(DatasetWrapper):
    def __init__(self, batch_size: int) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=transform)

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=transform)

        super().__init__(train_dataset, test_dataset, batch_size)
