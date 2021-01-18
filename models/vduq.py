from typing import Callable, Dict
from gpytorch.lazy.lazy_tensor import delazify
from models.model_params import GPParams, NNParams, TrainingParams
from datasets.activelearningdataset import ActiveLearningDataset, DatasetName
from models.model import UncertainModel, ModelWrapper
from models.model_params import ModelParams
from vduq.dkl import GP, DKL_GP
from vduq.wide_resnet import WideResNet
from vduq.small_resnet import MNISTResNet, PTMNISTResNet, CIFARResNet
from vduq.dkl import initial_values_for_GP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood
from marshmallow_dataclass import dataclass


import gpytorch
import torch


@dataclass
class vDUQParams(ModelParams):
    training_params: TrainingParams
    fe_params: NNParams
    gp_params: GPParams


class vDUQ(UncertainModel):
    fe_config = {
        DatasetName.mnist : [MNISTResNet, PTMNISTResNet],
        DatasetName.cifar10 : [CIFARResNet]
    }
    def __init__(self, params: vDUQParams, dataset: ActiveLearningDataset) -> None:
        # We pass the dataset so the model can properly initialise
        super().__init__()
        self.params = params
        self.reset(dataset)

        


    # To prepare for a new batch we need to update the size of the dataset and update the
    # classes who depend on the size of the training set
    def prepare(self, batch_size : int):
        self.num_data += batch_size
        if self.seperate_optimizers:
            self.variational_ngd_optimizer = gpytorch.optim.NGD(self.ngd_parameters, num_data= self.num_data,)
        
        self.elbo_fn = VariationalELBO(self.likelihood, self.model.gp,
                                       num_data=self.num_data)

    def get_eval_step(self):
        def eval_step(engine, batch):
            self.model.eval()
            self.likelihood.eval()

            x, y = batch
            if self.params.training_params.cuda:
                x, y = x.cuda(), y.cuda()

            with torch.no_grad():
                y_pred = self.model(x)

            return y_pred, y
        return eval_step

    def get_train_step(self):
        optimizer = self.optimizer
        if self.seperate_optimizers:
            ngd_optimizer = self.variational_ngd_optimizer

        def step(engine, batch):
            self.model.train()
            self.likelihood.train()

            optimizer.zero_grad()
            if self.seperate_optimizers:
                ngd_optimizer.zero_grad()
            x, y = batch
            if self.params.training_params.cuda:
                x, y = x.cuda(), y.cuda()

            y_pred = self.model(x)
            elbo = -self.elbo_fn(y_pred, y)
            elbo.backward()
            optimizer.step()
            if self.seperate_optimizers:
                ngd_optimizer.step()
            
            eig = torch.symeig(delazify(self.model.gp.covar_module(self.model.gp.inducing_points))).eigenvalues
            cond = eig.max() / eig.min()
            return {'loss': elbo.item(), 'cond': cond, 'min': eig.min()}
        return step

    def get_output_transform(self):
        def output_transform(output):
            y_pred, y = output

            # Sample softmax values independently for classification
            # at test time
            y_pred = y_pred.to_data_independent_dist()

            # The mean here is over likelihood samples
            y_pred = self.likelihood(y_pred).probs.mean(0)

            return y_pred, y
        return output_transform

    def get_model_params(self):
        return self.model_parameters

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self, optimizer):
        return self.scheduler

    def get_model(self):
        return self.model

    def reset(self, dataset : ActiveLearningDataset):
        params = self.params
        gp_params = params.gp_params
        fe_params = params.fe_params
        training_params = params.training_params

        train_dataset = dataset.get_train()
        self.num_data = len(train_dataset)

        # Initialise different feature extractors based on the dataset
        # We can have multiple configs per dataset
        # This allows us to compare various completely different architectures in a controlled way
        self.feature_extractor = vDUQ.fe_config[training_params.dataset][training_params.model_index](fe_params)

        init_inducing_points, init_lengthscale = initial_values_for_GP(
            train_dataset.dataset, self.feature_extractor,
            gp_params.n_inducing_points
        )

        gp = GP(
              num_outputs=gp_params.num_classes,
              initial_lengthscale=init_lengthscale,
              initial_inducing_points=init_inducing_points,
              separate_inducing_points=gp_params.separate_inducing_points,
              kernel=gp_params.kernel,
              ard=gp_params.ard,
              lengthscale_prior=gp_params.lengthscale_prior,
        )

        self.model = DKL_GP(self.feature_extractor, gp)

        self.likelihood = SoftmaxLikelihood(
            num_classes=gp_params.num_classes, mixing_weights=False)

        # Move to GPU if we can
        if self.params.training_params.cuda:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            init_inducing_points = init_inducing_points.cuda()


        self.model_parameters = [
            {"params": self.model.feature_extractor.parameters(),
                "lr": training_params.optimizers.optimizer},
            {"params": self.likelihood.parameters(
            ), "lr": training_params.optimizers.optimizer},
        ]

        # The ability to use split optimizers in vDUQ for the variational parameters 
        self.seperate_optimizers = (params.training_params.optimizers.var_optimizer is not None)
        if self.seperate_optimizers:
            self.ngd_parameters = [{
                "params": self.model.gp.parameters(),
                "lr": training_params.optimizers.optimizer}
            ]
            self.variational_ngd_optimizer = gpytorch.optim.NGD(self.ngd_parameters, num_data=self.num_data)

        else:
            self.model_parameters.append(
                 {"params": self.model.gp.parameters(
                ), "lr": training_params.optimizers.optimizer}
            )

        self.optimizer = torch.optim.SGD(
            self.model_parameters, momentum=0.9, weight_decay=fe_params.weight_decay
        )

        milestones = [60, 120, 160]

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.2
        )

        
        self.elbo_fn = VariationalELBO(self.likelihood, self.model.gp,
                                       num_data=self.num_data)

    def get_loss_fn(self):
        return lambda x,y : -self.elbo_fn(x,y)

    def get_training_params(self):
        return self.params.training_params

    def sample(self):
        return None

    def get_training_log_hooks(self) -> Dict[str, Callable[[Dict[str, float]], float]]:
        return {
            'cond': lambda x : x['cond'],
            'min' : lambda x : x['min'],
            'loss' : lambda x : x['loss'] 
        }
    
    def get_test_log_hooks(self) -> Dict[str, Callable[[Dict[str, float]], float]]:
        return {
            'accuracy' : self.get_output_transform(),
            'loss' : lambda x : x
        }

