from params.params import Params
from marshmallow_dataclass import dataclass


@dataclass
class DatasetParams(Params):
    path: str
    batch_size: int
    num_repetitions: int = 1
