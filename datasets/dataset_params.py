from typing import List, Optional, Tuple
from params.params import Params
from dataclasses import dataclass


@dataclass
class DatasetParams(Params):
    path: str
    batch_size: int
    class_weighting: tuple = tuple([])
    num_repetitions: int = 1
    smoke_test: bool = False
    classes : str = ','.join([str(i) for i in range(0,10)])
