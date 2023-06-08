__all__ = [
    "CustomTensorDataset",
    "LearnerParams",
    "ListOrInt",
    "OptInt",
    "OptStr",
    "OptTensor",
    "PathOrStr",
    "TensorGroup",
    "TorchOrNp"
]

from argparse import Namespace
from enum import Enum
import dataclasses
from pathlib import Path
from typing import Any, Callable, List, NoReturn, Optional, Set, Tuple, Union

import numpy as np
# from pandas import DataFrame

# from fastai.basic_data import DataBunch
from torch import Tensor
from torch.utils.data import Dataset

OptBool = Optional[bool]
OptCallable = Optional[Callable]
OptDict = Optional[dict]
OptFloat = Optional[float]
OptInt = Optional[int]
OptListInt = Optional[List[int]]
OptListStr = Optional[List[str]]
OptNamespace = Optional[Namespace]
OptStr = Optional[str]
OptTensor = Optional[Tensor]

ListOrInt = Union[int, List[int]]
SetListOrInt = Union[int, Set[int], List[int]]
SetOrList = Union[List[Any], Set[Any]]

PathOrStr = Union[Path, str]

TensorTuple = Tuple[Tensor, Tensor]
TorchOrNp = Union[Tensor, np.ndarray]


@dataclasses.dataclass
class TensorGroup:
    r""" Encapsulates a group of tensors used by the learner """
    tr_x: Optional[Tensor] = None
    tr_y: Optional[Tensor] = None
    tr_ids: Optional[Tensor] = None

    bd_x: Optional[Tensor] = None
    bd_y: Optional[Tensor] = None
    bd_ids: Optional[Tensor] = None

    val_x: Optional[Tensor] = None
    val_y: Optional[Tensor] = None
    val_ids: Optional[Tensor] = None

    targ_x: Optional[Tensor] = None
    targ_y: Optional[Tensor] = None  # Actual y value
    targ_ids: Optional[Tensor] = None

    te_cl_x: Optional[Tensor] = None
    te_cl_y: Optional[Tensor] = None
    te_cl_ids: Optional[Tensor] = None

    te_bd_x: Optional[Tensor] = None
    te_bd_y: Optional[Tensor] = None
    te_bd_ids: Optional[Tensor] = None


@dataclasses.dataclass(order=True)
class LearnerParams:
    r""" Learner specific parameters """
    class Attribute(Enum):
        LEARNING_RATE = "lr"
        WEIGHT_DECAY = "wd"

        # NUM_FF_LAYERS = "num_ff_layers"
        # NUM_SIGMA_LAYERS = "num_sigma_layers"

    learner_name: str

    lr: float = None
    wd: float = None

    # num_ff_layers: int = None
    # num_sigma_layers: int = None

    def set_attr(self, attr_name: str, value: Union[int, float]) -> NoReturn:
        r""" Enhanced set attribute method that has enhanced checking """
        try:
            # Allow short field name or longer attribute name
            attr_name = attr_name.lower()
            self.__getattribute__(attr_name)
        except AttributeError:
            try:
                attr_name = self.Attribute[attr_name.upper()].value
                self.__getattribute__(attr_name)
            except KeyError:
                raise AttributeError(f"No attribute \"{attr_name}\"")

        for field in dataclasses.fields(self):
            if field.name == attr_name:
                break
        else:
            raise ValueError(f"Cannot find field \"{attr_name}\"")

        assert isinstance(value, field.type), "Type mismatch when setting"
        self.__setattr__(attr_name, value)

    def get_attr(self, attr_name: str) -> Optional[Union[int, float]]:
        r""" Attribute accessor with more robust handling of attribute name """
        attr_name = attr_name.lower()
        try:
            return self.__getattribute__(attr_name)
        except AttributeError:
            raise AttributeError("No attribute \"attr_name\"")


class CustomTensorDataset(Dataset):
    r""" TensorDataset with support of transforms. """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = [tensor.clone() for tensor in tensors]
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)
        return tuple([x] + [tens[index] for tens in self.tensors[1:]])

        # y = self.tensors[1][index]

        # return x, y

    def __len__(self):
        return self.tensors[0].size(0)

    def set_transform(self, transform) -> NoReturn:
        r""" Change the transform for the dataset """
        self.transform = transform
