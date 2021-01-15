from abc import ABC, abstractmethod


class ModelParams(ABC):
    @abstractmethod
    def toDict(self):
        pass


class TrainingParams(ModelParams):
    def __init__(self, dict) -> None:
        super().__init__()
        self.epochs = dict['epochs']
        self.dataset = dict['dataset']
        self.cuda = dict['cuda']

    def toDict(self) -> str:
        pass


class GPParams(ModelParams):
    def __init__(self, dict) -> None:
        super().__init__()
        self.n_inducing_points = dict['n_inducing_points']
        self.num_classes = dict['num_classes']
        self.separate_inducing_points = dict['separate_inducing_points']
        self.kernel = dict['kernel']
        self.ard = dict['ard']
        self.lengthscale_prior = dict['lengthscale_prior']

    def toDict(self) -> str:
        pass


class NNParams(ModelParams):
    def __init__(self, dict) -> None:
        super().__init__()
        self.spectral_normalization = dict['spectral_normalization']
        self.dropout_rate = dict['dropout_rate']
        self.coeff = dict['coeff']
        self.learning_rate = dict['learning_rate']
        self.n_power_iterations = dict['n_power_iterations']
        self.batchnorm_momentum = dict['batchnorm_momentum']
        self.weight_decay = dict['weight_decay']

    def toDict(self) -> str:
        pass

