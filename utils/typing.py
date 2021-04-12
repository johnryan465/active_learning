if False:
    import torch
    from typing import Annotated
    # datapoints = channels = x = y = num_features = type(None)
    TensorType = type(Annotated[torch.Tensor, ...])
else:
    from torchtyping import TensorType, patch_typeguard
    patch_typeguard()