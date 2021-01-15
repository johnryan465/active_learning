from models.driver import Driver
from models.bnn import BayesianMNIST
from models.dnn import DNN
from models.dnn import DNNParams
from models.vduq import vDUQ
from models.vduq import vDUQParams
from params.params import GPParams, NNParams, TrainingParams

from datasets.mnist import MNIST
from methods.random import Random
from methods.BALD import BALD

# from methods.BatchBALD import BatchBALD
import torch

use_cuda = torch.cuda.is_available()

flag = False
bs = 512
epochs = 60

training_params = TrainingParams({
    'dataset': 'MNIST',
    'batch_size': bs,
    'epochs': epochs,
    'cuda': use_cuda
})

gp_params = GPParams({
    'kernel': 'RBF',
    'num_classes': 10,
    'ard': None,
    'n_inducing_points': 10,
    'lengthscale_prior': False,
    'separate_inducing_points': False
})

nn_params = NNParams({
    'spectral_normalization': True,
    'dropout_rate': 0.5,
    'coeff': 0.9,
    'n_power_iterations': 1,
    'batchnorm_momentum': 0.01,
    'weight_decay': 5e-4,
    'learning_rate': 0.01
})


dataset = MNIST(bs)
method = Random(bs,1,bs*60)
method.initialise(dataset)

if flag:
    # model = BayesianMNIST(nn_params, training_params)
    model = DNN(
        DNNParams(training_params, nn_params)
    )
else:
    params = vDUQParams(
        training_params,
        nn_params,
        gp_params
    )
    model = vDUQ(params, dataset)






if __name__ == "__main__":
    method.initialise(dataset)
    while(not method.complete()):
        # model.reset()
        Driver.train(model, dataset)
        # Driver.test(model, dataset)
        method.acquire(model, dataset)
        model.prepare(bs)