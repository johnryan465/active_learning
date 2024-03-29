from typing import Any, Dict
from uncertainty.multivariate_normal import MultitaskMultivariateNormalType

from gpytorch.lazy.cat_lazy_tensor import CatLazyTensor
from toma import toma
from tqdm.std import tqdm
from typeguard import typechecked

from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal

from uncertainty.fixed_dropout import BayesianModule
from gpytorch.lazy.lazy_tensor import delazify
from models.model_params import GPParams, NNParams
from datasets.activelearningdataset import ActiveLearningDataset, DatasetName
from models.model import UncertainModel
from models.training import TrainingParams
from models.model_params import ModelWrapperParams
from vduq.dkl import GP, DKL_GP
from models.mninst_base_models import MNISTResNet, PTMNISTResNet
from models.cifar_base_models import CIFARResNet
from vduq.dkl import initial_values_for_GP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood
from dataclasses import dataclass
from ignite.metrics.confusion_matrix import ConfusionMatrix

import gpytorch
import torch
from torch.utils.data import TensorDataset
from utils.typing import TensorType

@dataclass
class vDUQParams(ModelWrapperParams):
    fe_params: NNParams
    gp_params: GPParams


class vDUQ(UncertainModel):
    model : DKL_GP
    fe_config = {
        DatasetName.mnist: [MNISTResNet, PTMNISTResNet],
        DatasetName.cifar10: [CIFARResNet]
    }

    def __init__(self, model_params: vDUQParams, training_params: TrainingParams, dataset: ActiveLearningDataset) -> None:
        # We pass the dataset so the model can properly initialise
        super().__init__()
        self.params = model_params
        self.training_params = training_params
        self.initialize(dataset)

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
            elbo = -self.elbo_fn(y_pred, y) #type: ignore
            elbo.backward()
            optimizer.step()
            if self.seperate_optimizers:
                ngd_optimizer.step()

            # eig = torch.symeig(delazify(self.model.gp.covar_module(self.model.gp.inducing_points))).eigenvalues
            # cond = eig.max() / eig.min()
            return {'loss': elbo.item(), 'scale_norm': torch.mean(self.model.gp.covar_module.outputscale), 'mean': torch.mean(self.model.gp.mean_module.constant)}
        return step

    def get_output_transform(self):
        def output_transform(output):
            y_pred, y = output

            # Sample softmax values independently for classification
            # at test time
            y_pred = y_pred.to_data_independent_dist()

            # The mean here is over likelihood samples
            y_pred = self.likelihood(y_pred).probs.mean(0) #type: ignore
            return y_pred, y
        return output_transform

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self, optimizer):
        return self.scheduler

    def get_model(self):
        return self.model

    def get_num_cats(self) -> int:
        return 10

    def initialize(self, dataset: ActiveLearningDataset) -> None:
        params = self.params
        gp_params = params.gp_params
        fe_params = params.fe_params
        training_params = self.training_params

        train_dataset = dataset.get_train()
        self.num_data = len(train_dataset)
        num_data = self.num_data

        # Initialise different feature extractors based on the dataset
        # We can have multiple configs per dataset
        # This allows us to compare various completely different architectures in a controlled way
        feature_extractor = vDUQ.fe_config[training_params.dataset][params.model_index](fe_params)

        x_ = torch.stack([x[0] for x in dataset.get_train_tensor()])

        init_inducing_points, init_lengthscale = initial_values_for_GP(
            x_, feature_extractor,
            gp_params.n_inducing_points
        )

        # The ability to use split optimizers in vDUQ for the variational parameters
        self.seperate_optimizers = (training_params.optimizers.var_optimizer > 0)

        ard = gp_params.ard if gp_params.ard != -1 else None
        gp = GP(
              num_outputs=gp_params.num_classes,
              initial_lengthscale=init_lengthscale,
              initial_inducing_points=init_inducing_points,
              separate_inducing_points=gp_params.separate_inducing_points,
              kernel=gp_params.kernel,
              ard=ard,
              lengthscale_prior=gp_params.lengthscale_prior,
              var_dist="triangular" if self.seperate_optimizers else "default"
        )

        self.model = DKL_GP(feature_extractor, gp)

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
            self.variational_ngd_optimizer = gpytorch.optim.NGD(self.ngd_parameters, num_data=num_data)

        else:
            self.model_parameters.append({
                    "params": self.model.gp.parameters(),
                    "lr": training_params.optimizers.optimizer
                }
            )

        self.optimizer = torch.optim.Adam(
            self.model_parameters,
            lr=training_params.optimizers.optimizer
        )

        if num_data < 40:
            milestones = [ i * 8 for i in range(0,3)]
        else:
            milestones = [ i * int(num_data  / 5) for i in range(0,3)]

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=1
        )

        self.elbo_fn = VariationalELBO(self.likelihood, self.model.gp, num_data=num_data)


    def get_loss_fn(self):
        return lambda x, y: -self.elbo_fn(x, y) #type: ignore

    def get_training_params(self):
        return self.training_params

    # Depending on whether or not the feature extractor can be sampled from
    # we need to sample from the feature extractor and the GP
    # or just the GP
    def sample(self, input: torch.Tensor, samples: int) -> torch.Tensor:
        if isinstance(self.model.feature_extractor, BayesianModule):
            # We need to sample the feature extractor and the GP
            # We can either independently sample both
            # Or sample the fe k times
            # For each sample of the fe we sample the GP samples/k times
            # Requires sample which is a composite and a ratio
            fe_sampled = self.model.feature_extractor.forward(input, samples)
            flat = BayesianModule.flatten_tensor(fe_sampled)
            out = self.model.gp(flat)
            out = out.sample(torch.Size((1,)))
            out = out.permute(1, 0, 2)
            return self.likelihood(out).probs #type: ignore
        else:
            flat = self.model.feature_extractor.forward(input)
            out = self.model.gp(flat)
            out = out.sample(torch.Size((samples,)))
            out = out.permute(1, 0, 2)
            return self.likelihood(out).probs #type: ignore

    def get_training_log_hooks(self):
        return {
            'mean': lambda x: x['mean'],
            'scale_norm': lambda x: x['scale_norm'],
            'loss': lambda x: x['loss']
        }

    def get_test_log_hooks(self):
        return {
            'confusion': ConfusionMatrix(10),
            'accuracy': self.get_output_transform(),
            'loss': lambda x: x
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "training_params": self.training_params,
            "model_params": self.params,
            "likelihood": self.likelihood.state_dict()
        }

    @classmethod
    def load_state_dict(cls, state: Dict[str, Any], dataset: ActiveLearningDataset) -> 'vDUQ':
        params = state['model_params']
        training_params = state['training_params']

        # We create the new object and then update the weights

        model = vDUQ(params, training_params, dataset)
        model.model.load_state_dict(state['model'])
        model.likelihood.load_state_dict(state['likelihood'])
        return model

    def reset(self) -> None:
        return super().reset()

    @typechecked
    def get_features(self, inputs: TensorType["datapoints", "channels", "x", "y"]) -> TensorType["datapoints", "num_features"]:
        N = inputs.shape[0]
        feature_size = self.model.feature_extractor.features_size
        pool = torch.empty((N, feature_size))
        pbar = tqdm(total=N, desc="Feature Extraction", leave=False)
        @toma.execute.chunked(inputs, N)
        def compute(inputs, start: int, end: int):
            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                tmp = self.model.feature_extractor.forward(inputs).detach()
                pool[start:end].copy_(tmp, non_blocking=True)
            pbar.update(end - start)
        pbar.close()
        return pool


    @typechecked
    def get_gp_output(self, features: TensorType[ ..., "num_points", "num_features"]) -> MultitaskMultivariateNormalType:
        # We need to expand the dimensions of the features so we can broadcast with the GP
        if len(features.shape) > 2: # we have batches
            features = features.unsqueeze(-3)
        with torch.no_grad():
            dists = []
            N = features.shape[0]
            pbar = tqdm(total=N, desc="GP", leave=False)
            @toma.execute.chunked(features, N)
            def compute(features, start: int, end: int):
                if torch.cuda.is_available():
                    features = features.cuda()
                d = self.model.gp(features)
                dists.append(d)
                pbar.update(end - start)
            pbar.close()
            # We want to keep things off the GPU
            dist = MultitaskMultivariateNormalType.combine_mtmvns(dists)
            mean_cpu = dist.mean
            cov_cpu = dist.lazy_covariance_matrix
            if torch.cuda.is_available():
                if(isinstance(mean_cpu, CatLazyTensor)):
                    mean_cpu = mean_cpu.all_to("cpu")
                else:
                    mean_cpu = mean_cpu.cpu()

                if(isinstance(cov_cpu, CatLazyTensor)):
                    cov_cpu = cov_cpu.all_to("cuda")
                else:
                    cov_cpu = cov_cpu.cuda()

            return MultitaskMultivariateNormalType(mean=mean_cpu, covariance_matrix=cov_cpu)
