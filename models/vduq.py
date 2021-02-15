from typing import Callable, Dict


from uncertainty.fixed_dropout import BayesianModule
from gpytorch.lazy.lazy_tensor import delazify
from models.model_params import GPParams, NNParams
from datasets.activelearningdataset import ActiveLearningDataset, DatasetName
from models.model import UncertainModel
from models.training import TrainingParams
from models.model_params import ModelWrapperParams
from vduq.dkl import GP, DKL_GP
from vduq.small_resnet import BNNMNISTResNet, MNISTResNet, PTMNISTResNet, CIFARResNet
from vduq.dkl import initial_values_for_GP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood
from marshmallow_dataclass import dataclass


import gpytorch
import torch
from torch.utils.data import TensorDataset


@dataclass
class vDUQParams(ModelWrapperParams):
    fe_params: NNParams
    gp_params: GPParams


class vDUQ(UncertainModel):
    fe_config = {
        DatasetName.mnist: [MNISTResNet, PTMNISTResNet, BNNMNISTResNet],
        DatasetName.cifar10: [CIFARResNet]
    }

    def __init__(self, model_params: vDUQParams, training_params: TrainingParams, dataset: ActiveLearningDataset) -> None:
        # We pass the dataset so the model can properly initialise
        super().__init__()
        self.params = model_params
        self.training_params = training_params
        self.reset(dataset)

    # To prepare for a new batch we need to update the size of the dataset and update the
    # classes who depend on the size of the training set

    def prepare(self, batch_size: int):
        self.num_data += batch_size
        if self.seperate_optimizers:
            self.variational_ngd_optimizer = gpytorch.optim.NGD(self.ngd_parameters, num_data=self.num_data,)

        self.elbo_fn = VariationalELBO(self.likelihood, self.model.gp, num_data=self.num_data)

    def get_eval_step(self):
        def eval_step(engine, batch):
            self.model.eval()
            self.likelihood.eval()

            x, y = batch
            if self.training_params.cuda:
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
            if self.training_params.cuda:
                x, y = x.cuda(), y.cuda()

            # Get a a better estimate
            with gpytorch.settings.num_likelihood_samples(32):
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

    def reset(self, dataset: ActiveLearningDataset):
        params = self.params
        gp_params = params.gp_params
        fe_params = params.fe_params
        training_params = self.training_params

        train_dataset = dataset.get_train()
        self.num_data = len(train_dataset)

        # Initialise different feature extractors based on the dataset
        # We can have multiple configs per dataset
        # This allows us to compare various completely different architectures in a controlled way
        self.feature_extractor = vDUQ.fe_config[training_params.dataset][params.model_index](fe_params)

        dataset_list = list(iter(train_dataset))

        x_ = torch.cat([x[0] for x in dataset_list])
        y_ = torch.cat([x[1] for x in dataset_list])

        init_inducing_points, init_lengthscale = initial_values_for_GP(
            TensorDataset(x_, y_), self.feature_extractor,
            gp_params.n_inducing_points
        )

        # The ability to use split optimizers in vDUQ for the variational parameters
        self.seperate_optimizers = (training_params.optimizers.var_optimizer is not None)

        gp = GP(
              num_outputs=gp_params.num_classes,
              initial_lengthscale=init_lengthscale,
              initial_inducing_points=init_inducing_points,
              separate_inducing_points=gp_params.separate_inducing_points,
              kernel=gp_params.kernel,
              ard=gp_params.ard,
              lengthscale_prior=gp_params.lengthscale_prior,
              var_dist="triangular" if self.seperate_optimizers else "default"
        )

        self.model = DKL_GP(self.feature_extractor, gp)

        self.likelihood = SoftmaxLikelihood(
            num_classes=gp_params.num_classes, mixing_weights=False)

        # Move to GPU if we can
        if self.training_params.cuda:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            init_inducing_points = init_inducing_points.cuda()

        self.model_parameters = [
            {"params": self.model.feature_extractor.parameters(),
                "lr": training_params.optimizers.optimizer},
            {"params": self.likelihood.parameters(
            ), "lr": training_params.optimizers.optimizer},
        ]

        if self.seperate_optimizers:
            self.ngd_parameters = [{
                "params": self.model.gp.parameters(),
                "lr": training_params.optimizers.optimizer}
            ]
            self.variational_ngd_optimizer = gpytorch.optim.NGD(self.ngd_parameters, num_data=self.num_data)

        else:
            self.model_parameters.append({
                    "params": self.model.gp.parameters(),
                    "lr": training_params.optimizers.optimizer
                }
            )

        self.optimizer = torch.optim.SGD(
            self.model_parameters, momentum=0.9, weight_decay=fe_params.weight_decay
        )

        milestones = [60, 120, 160]

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=1
        )

        self.elbo_fn = VariationalELBO(self.likelihood, self.model.gp,
                                       num_data=self.num_data)

    def get_loss_fn(self):
        return lambda x, y: -self.elbo_fn(x, y)

    def get_training_params(self):
        return self.training_params

    # Depending on whether or not the feature extractor can be sampled from
    # we need to sample from the feature extractor and the GP
    # or just the GP
    def sample(self, input: torch.Tensor, samples: int) -> torch.Tensor:
        if isinstance(self.feature_extractor, BayesianModule):
            # We need to sample the feature extractor and the GP
            # We can either independently sample both
            # Or sample the fe k times
            # For each sample of the fe we sample the GP samples/k times
            # Requires sample which is a composite and a ratio
            fe_sampled = self.feature_extractor.forward(input, samples)
            flat = BayesianModule.flatten_tensor(fe_sampled)
            out = self.model.gp(flat)
            out = out.sample(torch.Size((1,)))
            out = out.permute(1, 0, 2)
            return self.likelihood(out).probs
        else:
            flat = self.feature_extractor.forward(input)
            out = self.model.gp(flat)
            out = out.sample(torch.Size((samples,)))
            out = out.permute(1, 0, 2)
            return self.likelihood(out).probs

    def get_training_log_hooks(self) -> Dict[str, Callable[[Dict[str, float]], float]]:
        return {
            'cond': lambda x: x['cond'],
            'min': lambda x: x['min'],
            'loss': lambda x: x['loss']
        }

    def get_test_log_hooks(self) -> Dict[str, Callable[[Dict[str, float]], float]]:
        return {
            'accuracy': self.get_output_transform(),
            'loss': lambda x: x
        }
