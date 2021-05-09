import torch
from typing import Annotated
from torchtyping import TensorType, patch_typeguard

if False:
    TensorType = type(Annotated[torch.Tensor, ...])
else:
    patch_typeguard()


