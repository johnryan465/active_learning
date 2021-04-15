from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from gpytorch.distributions.multivariate_normal import MultivariateNormal
import torch
from typing import Annotated, Any, Generic, List, NoReturn, Type, TypeVar
from torchtyping import TensorType, patch_typeguard
from torchtyping.utils import frozendict

import ast
import inspect
import types
import functools
from typing import List, Any
from typeguard import check_type
from torchtyping import patch_typeguard
from ast import Expr, Call, Name, Load, Constant, Subscript, Return, Load, fix_missing_locations
import copy
import __future__
import re
T = TypeVar("T")

# This is similar to torchtyping, with annotations but currently without any of the clever shape checking ... yet
class _DistributionTypeMeta(type):
    def __new__(cls, name, bases, dict, base_cls):
        new_cls = super().__new__(cls, name, bases, dict)
        new_cls.base_cls = base_cls # type: ignore
        return new_cls

    def __instancecheck__(cls, obj: Any) -> bool:
        return isinstance(obj, cls.base_cls) # type: ignore



def create_type(base_cls) -> Type:
    class DistributionType(Generic[T], metaclass=_DistributionTypeMeta, base_cls=base_cls):
        base_cls = None
        def __new__(cls, *args, **kwargs) -> NoReturn:
            raise RuntimeError(f"Class {cls.__name__} cannot be instantiated.")

        @staticmethod
        def _type_error(item: Any) -> NoReturn:
            raise TypeError(f"{item} not a valid type argument.")

        def __class_getitem__(cls, item: Any) -> Type[Annotated[T, ...]]:
            if isinstance(item, tuple):
                if len(item) == 0:
                    item = ((),)
            else:
                item = (item,)

            details = [item]
            pre_details = []

            details = tuple(pre_details + details)

            assert len(details) > 0

            # Frozen dict needed for Union[TensorType[...], ...], as Union hashes its
            # arguments.
            return Annotated[
                cls.base_cls, # type: ignore
                frozendict(
                    {"__torchtyping__": True, "details": details, "cls_name": cls.__name__}
                ),
            ]
    return DistributionType



patch_typeguard()


# This flag enables to toggle between dumb annotations which vscode can understand and the runtime checking
# Having it set to a constant lets the static analysiser select the code path designed for it
if False:
    # datapoints = channels = x = y = num_features = type(None)
    MultitaskMultivariateNormalType = type(Annotated[MultitaskMultivariateNormal, ...])
    MultivariateNormalType = type(Annotated[MultivariateNormal, ...])
    TensorType = type(Annotated[torch.Tensor, ...])
else:
    MultitaskMultivariateNormalType = create_type(MultitaskMultivariateNormal)
    MultivariateNormalType = create_type(MultivariateNormal)
    patch_typeguard()


