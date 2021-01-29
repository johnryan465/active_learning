import math
from uncertainty.fixed_dropout import BayesianModule
import torch

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.priors import SmoothedBoxPrior

from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

from sklearn import cluster


def initial_values_for_GP(train_dataset, feature_extractor, n_inducing_points):
    # Following Bradshaw 2017
    # https://gist.github.com/john-bradshaw/e6784db56f8ae2cf13bb51eec51e9057

    steps = 5
    idx = torch.randperm(len(train_dataset))[:1000].chunk(steps)
    print(len(train_dataset))
    print(idx)
    f_X_samples = []
    uncertain_fe = isinstance(feature_extractor, BayesianModule)

    with torch.no_grad():
        for i in range(steps):
            #print(train_dataset[0])
            # print(i)
            X_sample = torch.stack([train_dataset[j][0] for j in idx[i]])

            if torch.cuda.is_available():
                X_sample = X_sample.cuda()
                feature_extractor = feature_extractor.cuda()

            if uncertain_fe:
                f_X_samples.append(feature_extractor(X_sample, 1).squeeze().cpu())
            else:
                f_X_samples.append(feature_extractor(X_sample).cpu())

    f_X_samples = torch.cat(f_X_samples)

    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), n_inducing_points
    )
    initial_lengthscale = _get_initial_lengthscale(f_X_samples)

    return initial_inducing_points, initial_lengthscale


def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    print(f_X_sample.shape)
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
    )
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)
    return initial_inducing_points


def _get_initial_lengthscale(f_X_samples):
    if torch.cuda.is_available():
        f_X_samples = f_X_samples.cuda()

    initial_lengthscale = torch.pdist(f_X_samples).mean()

    return initial_lengthscale.cpu()


class GP(ApproximateGP):
    def __init__(
        self,
        num_outputs,
        initial_lengthscale,
        initial_inducing_points,
        separate_inducing_points=False,
        kernel="RBF",
        ard=None,
        lengthscale_prior=False,
    ):
        n_inducing_points = initial_inducing_points.shape[0]
        if separate_inducing_points:
            # Use independent inducing points per output GP
            initial_inducing_points = initial_inducing_points.repeat(
                num_outputs, 1, 1)

        if num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            batch_shape = torch.Size([])

        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self, initial_inducing_points, variational_distribution
        )

        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_outputs
            )

        super().__init__(variational_strategy)

        if lengthscale_prior:
            lengthscale_prior = SmoothedBoxPrior(
                math.exp(-1), math.exp(1), sigma=0.1)
        else:
            lengthscale_prior = None

        kwargs = {
            "ard_num_dims": ard,
            "batch_shape": batch_shape,
            "lengthscale_prior": lengthscale_prior,
        }

        if kernel == "RBF":
            kernel = RBFKernel(**kwargs)
        elif kernel == "Matern12":
            kernel = MaternKernel(nu=1 / 2, **kwargs)
        elif kernel == "Matern32":
            kernel = MaternKernel(nu=3 / 2, **kwargs)
        elif kernel == "Matern52":
            kernel = MaternKernel(nu=5 / 2, **kwargs)
        elif kernel == "RQ":
            kernel = RQKernel(**kwargs)
        else:
            raise ValueError("Specified kernel not known.")

        kernel.lengthscale = initial_lengthscale * torch.ones_like(
            kernel.lengthscale
        )

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(kernel, batch_shape=batch_shape)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return MultivariateNormal(mean, covar)

    @property
    def inducing_points(self):
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param


class DKL_GP(gpytorch.Module):
    def __init__(self, feature_extractor, gp):
        """
        This wrapper class is necessary because ApproximateGP (above) does
        some magic on the forward method which is not compatible with a
        feature_extractor.
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.uncertain_fe = isinstance(feature_extractor, BayesianModule)
        self.gp = gp

    def forward(self, x):
        if self.uncertain_fe:
            features = self.feature_extractor(x, 1).squeeze()
        else:
            features = self.feature_extractor(x)
        return self.gp(features)
