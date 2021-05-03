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




class MNISTResNet(FeatureExtractor):
    def __init__(self, nn_params: NNParams):
        super().__init__()
        channels = 1
        image_size = 28
        num_classes = None
        params = nn_params
        
        self.dropout_rate = params.dropout_rate
        self.image_size = image_size
        self.features_size = 256

        coeff = params.coeff
        batchnorm_momentum = params.batchnorm_momentum
        n_power_iterations = params.n_power_iterations
        spectral_normalization = params.spectral_normalization

        def wrapped_bn(num_features):
            if spectral_normalization:
                bn = SpectralBatchNorm2d(
                    num_features, coeff, momentum=batchnorm_momentum
                )
            else:
                bn = nn.BatchNorm2d(num_features, momentum=batchnorm_momentum)

            return bn

        self.wrapped_bn = wrapped_bn

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride, padding=None):
            if padding is None:
                padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size,
                             stride, padding, bias=False)

            if not spectral_normalization:
                return conv
            if kernel_size == 1:
                # use spectral norm fc, because bound are
                # tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff,
                                                n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                input_dim = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, input_dim, n_power_iterations
                )

            return wrapped_conv

        self.wrapped_conv = wrapped_conv

        strides = [1, 1, 2, 2, 2]
        nStages = 64 * np.cumprod(strides) #[64, 128, 256, 512, 1024]
        layer_size = [2, 2, 2, 2]
        input_sizes = image_size // np.cumprod(strides)

        n = 1  # layer_size
        num_blocks = len(layer_size)
        self.conv1 = self.wrapped_conv(image_size, 1, 64, 7, 2, padding=3)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # "layers" are layers in the resnet sense, collections of blocks
        # Layers are connected together with convolutions which reduce the spatial dimensions
        self.layers = nn.ModuleList()

        # for i in range(0, num_blocks):
        self.layers.append(self._layer(nStages[0: 2], n, strides[1], 14))
        self.layers.append(self._layer(nStages[1: 3], n, strides[2], 14))
        self.layers.append(self._layer(nStages[2: 4], n, strides[3], 7))

        # print(self.layers)
        self.bn1 = self.wrapped_bn(nStages[0])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes
        if num_classes is not None:
            self.linear = nn.Linear(nStages[num_blocks - 1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def _layer(self, channels, num_blocks, stride, input_size):
        # We have a possibly different stride as the first one
        # as we could be at the beginning of a resnet layer where we are projecting
        # down the number of spatial dimesions
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        in_c, out_c = channels

        for stride in strides:
            layers.append(
                WideBasic(
                    self.wrapped_conv,
                    self.wrapped_bn,
                    input_size,
                    in_c,
                    out_c,
                    stride,
                    self.dropout_rate,
                )
            )
            in_c = out_c
            input_size = input_size // stride

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        # out = self.maxpool(out)
        for layer in self.layers:
            out = layer(out)
        out = self.avgpool(out)
        out = out.flatten(1)

        if self.num_classes is not None:
            out = self.linear(out)

        return out