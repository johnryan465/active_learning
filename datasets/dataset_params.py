from abc import ABC, abstractmethod
from params.params import Params
from typing import List, Union
from datasets.activelearningdataset import DatasetName
from marshmallow_dataclass import dataclass


@dataclass
class DatasetParams(Params):
    batch_size: int