from dataclasses import fields
from typing import Any
from params.registered import Registered
from params.params import Params
import os.path
import json
import csv
from ignite.metrics.loss import Loss
from typing import Callable, Dict, Sequence, Tuple, Union, cast

import torch

from ignite.metrics.metric import reinit__is_reduced


class IO:
    @staticmethod
    def parseParams(cls: type, dic: dict) -> Any:
        kwds = {}
        for field in fields(cls):
            a = dic.get(field.name)
            if issubclass(field.type, Params):
                for option in Registered.types[field.type.name()]:
                    if option.name() == dic[field.name]['__name__']:
                        a = IO.parseParams(option, dic[field.name])

            kwds[field.name] = a

        return cls(**kwds)

    @staticmethod
    def create_directory(path: str) -> None:
        # remove the file name
        directory = "/".join(path.split('/')[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def dict_to_file(dic: dict, file_name: str) -> None:
        IO.create_directory(file_name)
        with open(file_name, 'w') as outfile:
            json.dump(dic, outfile, indent=4)

    @staticmethod
    def load_dict_from_file(file_name: str) -> dict:
        with open(file_name) as json_file:
            data = json.load(json_file)
        return data

    @staticmethod
    def file_exists(file_name: str) -> bool:
        return os.path.isfile(file_name)

    @staticmethod
    def dict_to_csv(dic: dict, file_name: str) -> None:
        IO.create_directory(file_name)
        keys = dic[0].keys()
        with open(file_name, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(dic)


# We now cant use the default loss in ignite as it attempts to detach within the the update
# function, meaning we can't simply pass in a Distribuiton as our prediction.
class VariationalLoss(Loss):
    def __init__(
        self,
        loss_fn: Callable,
        output_transform: Callable = lambda x: x,
        batch_size: Callable = lambda x: len(x),
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(VariationalLoss, self).__init__(loss_fn, output_transform, batch_size, device)

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, Dict]]) -> None:
        if len(output) == 2:
            y_pred, y = cast(Tuple[torch.Tensor, torch.Tensor], output)
            kwargs = {}  # type: Dict
        else:
            y_pred, y, kwargs = cast(Tuple[torch.Tensor, torch.Tensor, Dict], output)
        average_loss = self._loss_fn(y_pred, y.detach(), **kwargs)

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        n = self._batch_size(y)
        self._sum += average_loss.to(self._device) * n
        self._num_examples += n
