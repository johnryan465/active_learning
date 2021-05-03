# A resnet for MNIST as a sanity check comparison to the custom one below



import numpy as np
from vduq.layers import spectral_norm_conv, spectral_norm_fc
from vduq.layers.spectral_batchnorm import SpectralBatchNorm2d
from vduq.wide_resnet import WideBasic
from uncertainty.fixed_dropout import BayesianModule, ConsistentMCDropout, ConsistentMCDropout2d
from models.model_params import NNParams
from models.base_models import FeatureExtractor
from torchvision.models.resnet import ResNet, BasicBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

class PTMNISTResNet(ResNet, FeatureExtractor):
    def __init__(self, params: NNParams):
        super(PTMNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.fc = nn.Identity()

    def forward(self, x):
        return super(PTMNISTResNet, self).forward(x)


class CNNMNIST(FeatureExtractor):
    def __init__(self, params: NNParams):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1(input), 2))
        input = F.relu(F.max_pool2d(self.conv2(input), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1(input))
        input = self.fc2(input)
        # input = F.log_softmax(input, dim=1)
        return input


# class BNNMNISTResNet(BayesianModule, FeatureExtractor):
#     def __init__(self, params: NNParams):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
#         self.conv1_drop = ConsistentMCDropout2d()
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
#         self.conv2_drop = ConsistentMCDropout2d()
#         self.fc1 = nn.Linear(1024, 128)
#         self.fc1_drop = ConsistentMCDropout()
#         self.fc2 = nn.Linear(128, 128)

#     def mc_forward_impl(self, input: torch.Tensor):
#         input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
#         input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
#         input = input.view(-1, 1024)
#         input = F.relu(self.fc1_drop(self.fc1(input)))
#         input = self.fc2(input)
#         # input = F.log_softmax(input, dim=1)
#         return input



# We need to reduce the number of parameters in this model
class MNISTResNet(FeatureExtractor):
    def __init__(self, nn_params: NNParams):
        super().__init__()
        depth = 28
        channels = 1
        input_size = 28
        num_classes = nn_params.num_classes
        params = nn_params
        self.num_features = 1024
        
        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"

        self.dropout_rate = nn_params.dropout_rate
        spectral_normalization = nn_params.spectral_normalization
        coeff = nn_params.coeff
        n_power_iterations = nn_params.n_power_iterations
        batchnorm_momentum = nn_params.batchnorm_momentum

        def wrapped_bn(num_features):
            if spectral_normalization:
                bn = SpectralBatchNorm2d(
                    num_features, coeff, momentum=batchnorm_momentum
                )
            else:
                bn = nn.BatchNorm2d(num_features, momentum=batchnorm_momentum)

            return bn

        self.wrapped_bn = wrapped_bn

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            if kernel_size == 3:
                padding = 1
            elif kernel_size == 5:
                padding = 2
            else:
                padding = 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_normalization:
                return conv

            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                input_dim = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, input_dim, n_power_iterations
                )

            return wrapped_conv

        self.wrapped_conv = wrapped_conv



        self.conv1 = wrapped_conv(input_size, channels, 32, 5, 1)
        self.shortcut1 = wrapped_conv(input_size, channels, 32, 1, 1)
        self.layer1 = wrapped_conv(input_size, 32, 64, 5, 1)
        self.shortcut2 = wrapped_conv(input_size, 32, 64, 1, 1)
        self.bn1 = self.wrapped_bn(64)

        self.num_classes = num_classes
        if num_classes is not None:
            self.linear = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(x.shape)
        out1 = self.conv1(x)
        out1 += self.shortcut1(x)
        out2 = self.layer1(out1)
        out2 += self.shortcut2(out1)
        out2 = F.relu(self.bn1(out2))
        out = F.avg_pool2d(out2, 7)
        out = out.flatten(1)

        if self.num_classes is not None:
            out = self.linear(out)
        return out
