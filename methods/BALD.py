from dataclasses import dataclass

from models.model import UncertainModel
from datasets.activelearningdataset import ActiveLearningDataset
from methods.method import UncertainMethod, Method
from methods.method_params import MethodParams
from batchbald_redux.batchbald import get_bald_batch
from datasets.activelearningdataset import DatasetUtils
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
import torch


@dataclass
class BALDParams(MethodParams):
    samples: int


class BALD(UncertainMethod):
    def __init__(self, params: BALDParams) -> None:
        super().__init__()
        self.params = params
        self.current_aquisition = 0

    def acquire(self, model: UncertainModel,
                dataset: ActiveLearningDataset, tb_logger: TensorboardLogger) -> None:
        if isinstance(model, vDUQ):
            pass
        else:
            probs = []
            for x, _ in dataset.get_pool():
                if torch.cuda.is_available():
                    x = x.cuda()
                probs_ = model.sample(x, self.params.samples).detach().clone()
                probs.append(probs_)

            probs = torch.cat(probs, dim=0)
            batch = get_bald_batch(probs, self.params.aquisition_size)
            Method.log_batch(batch.indices, tb_logger, self.current_aquisition)
            dataset.move(batch.indices)
            self.current_aquisition += 1

    def initialise(self, dataset: ActiveLearningDataset) -> None:
        DatasetUtils.balanced_init(dataset, self.params.initial_size)

    def complete(self) -> bool:
        return self.current_aquisition >= self.params.max_num_aquisitions
