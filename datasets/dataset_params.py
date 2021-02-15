from params.params import Params
from marshmallow_dataclass import dataclass


@dataclass
class DatasetParams(Params):
    batch_size: int
