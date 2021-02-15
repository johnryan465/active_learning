import torchvision
import torchvision.transforms as transforms
from .activelearningdataset import DatasetName, DatasetWrapper
from .dataset_params import DatasetParams


class MNIST(DatasetWrapper):
    def __init__(self, config: DatasetParams) -> None:
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))])

        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True,
            transform=transform)

        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True,
            transform=transform)
        super().__init__(train_dataset, test_dataset, config.batch_size)

    def get_name(self) -> DatasetName:
        return DatasetName.mnist
