from gpytorch.lazy.lazy_tensor import delazify
from params.params import GPParams, NNParams, TrainingParams
from datasets.activelearningdataset import ActiveLearningDataset
from models.model import UncertainModel, ModelWrapper
from params.params import ModelParams
from vduq.dkl import GP, DKL_GP
from vduq.wide_resnet import WideResNet
from vduq.small_resnet import MnistResNet
from vduq.dkl import initial_values_for_GP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood
import gpytorch
import torch


class vDUQParams(ModelParams):
    def __init__(self, training_params: TrainingParams,
                 fe_params: NNParams, gp_params: GPParams) -> None:
        super().__init__()
        self.training_params = training_params
        self.fe_params = fe_params
        self.gp_params = gp_params

    def toDict(self) -> str:
        pass


class vDUQ(UncertainModel):
    def __init__(self, params: vDUQParams, dataset: ActiveLearningDataset) -> None:
        # We pass the dataset so the model can properly initialise
        super().__init__()
        self.params = params
        train_dataset = dataset.get_train()

        params = self.params
        gp_params = params.gp_params
        fe_params = params.fe_params

        # Initialise different feature extractors based on the dataset
        if self.params.training_params.dataset == "MNIST":
            self.feature_extractor = MnistResNet(
                spectral_normalization=fe_params.spectral_normalization,
                coeff=0.9,
                batchnorm_momentum=0.9,
                n_power_iterations=1,
                dropout_rate=0.3
            )
            """ WideResNet(
                spectral_normalization=fe_params.spectral_normalization,
                dropout_rate=fe_params.dropout_rate,
                coeff=fe_params.coeff,
                channels=1,
                image_size=28,
                depth=10,
                n_power_iterations=fe_params.n_power_iterations,
                batchnorm_momentum=fe_params.batchnorm_momentum,
            ) """
            
        elif self.params.training_params.dataset == "CIFAR10":
            self.feature_extractor = WideResNet(
                spectral_normalization=fe_params.spectral_normalization,
                dropout_rate=fe_params.dropout_rate,
                coeff=fe_params.coeff,
                channels=3,
                image_size=32,
                n_power_iterations=fe_params.n_power_iterations,
                batchnorm_momentum=fe_params.batchnorm_momentum,
            )
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

        if self.params.training_params.cuda:
            self.model = self.model.cuda()

        self.likelihood = SoftmaxLikelihood(
            num_classes=gp_params.num_classes, mixing_weights=False)

        if self.params.training_params.cuda:
            self.likelihood = self.likelihood.cuda()

        if self.params.training_params.cuda:
            init_inducing_points = init_inducing_points.cuda()

        self.model_parameters = [
            {"params": self.model.feature_extractor.parameters(),
                "lr": fe_params.learning_rate},
            {"params": self.likelihood.parameters(
            ), "lr": fe_params.learning_rate},
        ]

        self.ngd_parameters = [
            {"params": self.model.gp.parameters(
            ), "lr": fe_params.learning_rate},
        ]

        self.optimizer = torch.optim.SGD(
            self.model_parameters, momentum=0.9, weight_decay=fe_params.weight_decay
        )

        milestones = [60, 120, 160]

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.2
        )
        self.num_data = len(train_dataset)
        self.variational_ngd_optimizer = gpytorch.optim.NGD(self.ngd_parameters, num_data= self.num_data, lr=0.1)

        
        self.elbo_fn = VariationalELBO(self.likelihood, self.model.gp,
                                       num_data=self.num_data)




    def prepare(self, batch_size : int):
        self.num_data += batch_size
        self.variational_ngd_optimizer = gpytorch.optim.NGD(self.ngd_parameters, num_data= self.num_data, lr=0.1)
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
        ngd_optimizer = self.variational_ngd_optimizer
        def step(engine, batch):
            self.model.train()
            self.likelihood.train()

            optimizer.zero_grad()
            ngd_optimizer.zero_grad()
            x, y = batch
            if self.params.training_params.cuda:
                x, y = x.cuda(), y.cuda()

            y_pred = self.model(x)
            elbo = -self.elbo_fn(y_pred, y)

            elbo.backward()
            optimizer.step()
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

    def reset(self):
        self.__init__(self, self.params)

    def get_loss_fn(self):
        return lambda x,y : -self.elbo_fn(x,y)

    def get_training_params(self):
        return self.params.training_params

    def sample(self):
        return None
