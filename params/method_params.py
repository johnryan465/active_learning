from abc import ABC, abstractmethod
from params.params import Params
from typing import List, Union
from dataclasses import dataclass

@dataclass
class MethodParams(Params):
    pass


@dataclass
class RandomParams(MethodParams):
    batch_size : int
