
from models.model_params import NNParams
from models.base_models import FeatureExtractor
from vduq.wide_resnet import WideResNet

class CIFARResNet(WideResNet, FeatureExtractor):
    def __init__(self, params: NNParams):
        super(CIFARResNet, self).__init__(
            spectral_normalization=params.spectral_normalization,
            dropout_rate=params.dropout_rate,
            coeff=params.coeff,
            channels=3,
            image_size=32,
            n_power_iterations=params.n_power_iterations,
            batchnorm_momentum=params.batchnorm_momentum,
        )