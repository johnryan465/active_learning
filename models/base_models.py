import torch.nn as nn
import torch.nn.functional as F



class FeatureExtractor(nn.Module):
    pass

class LogSoftmaxModel(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor):
        super(LogSoftmaxModel, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        return F.log_softmax(self.feature_extractor(x), dim=1)