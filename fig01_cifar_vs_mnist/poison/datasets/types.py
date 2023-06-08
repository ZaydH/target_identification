__all__ = [
    "BaseFFModule",
    "LearnerModule",
    "NEG_LABEL",
    "POS_LABEL",
    "PoisonDataset",
    "PoisonLearner",
    "SmartLinear",
    "ViewTo1D",
]

import abc
import collections
import copy
from enum import Enum
import logging
from typing import List, NoReturn, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .. import _config as config

DatasetParams = collections.namedtuple("DatasetParams", "name dim n_classes")
CIFAR_DIM = [3, 32, 32]
MNIST_DIM = [1, 28, 28]

NEG_LABEL = 0
POS_LABEL = +1


# noinspection PyPep8Naming
class PoisonDataset(Enum):
    r""" Valid datasets for testing """
    CIFAR10 = DatasetParams("CIFAR10", CIFAR_DIM, 10)

    MNIST = DatasetParams("MNIST", MNIST_DIM, 10)

    def is_cifar(self) -> bool:
        r""" Returns \p True if dataset is a CIFAR dataset """
        return self == self.CIFAR10

    def is_mnist(self) -> bool:
        r""" Returns \p True if dataset is a MNIST or MNIST-variant dataset """
        return self == self.MNIST


# noinspection PyPep8Naming
class LearnerModule(nn.Module):
    # TORCH_DEVICE = None

    def __init__(self):
        super().__init__()
        self._model = None

    def forward(self, x: Tensor) -> Tensor:
        # if self.TORCH_DEVICE is not None: x = x.to(self.TORCH_DEVICE)
        assert self.is_model_set(), "Model not yet set"
        dec_scores = self._model(x).squeeze(dim=1)
        return dec_scores

    def predict(self, x) -> Tensor:
        r""" Provides a class prediction, i.e., positive or negative """
        y_hat = self.forward(x)
        if config.DATASET.is_cifar() or config.DATASET.is_mnist():
            lbls = torch.sign(y_hat).long()
            lbls[lbls == -1] = NEG_LABEL
        else:
            raise ValueError("Unknown dataset.  Do not know how to predict")

        # if self.TORCH_DEVICE is not None:
        #     lbls.to(self.TORCH_DEVICE)
        return lbls

    def is_model_set(self) -> bool:
        r""" Standardizes checking whether a model has been set"""
        return self._model is not None

    def set_model(self, module: nn.Module) -> None:
        r""" Sets the underlying module used """
        self._model = copy.deepcopy(module)

    def get_module(self) -> nn.Module:
        r""" Accessor for the module """
        assert self.is_model_set(), "Model not yet set"
        return self._model


class ViewModule(nn.Module):
    r""" General view layer to flatten to any output dimension """
    def __init__(self, d_out: List[int]):
        super().__init__()
        self._d_out = tuple(d_out)

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        # noinspection PyUnresolvedReferences
        return x.reshape((x.shape[0], *self._d_out))


class ViewTo1D(ViewModule):
    r""" View layer simplifying to specifically a single dimension """
    def __init__(self):
        super().__init__([-1])


class BaseFFModule(LearnerModule):

    class Config:
        r""" Configuration settings for the 20 newsgroups learner """
        # FF_HIDDEN_DEPTH = 2
        FF_HIDDEN_DIM = 300
        FF_ACTIVATION = nn.ReLU

    def __init__(self, x: Tensor, n_class: int, num_hidden_layers: int):
        super().__init__()

        self._model.add_module("View1D", ViewTo1D())
        in_dim = x[0].numel()

        self._n_class = n_class
        self._n_hidden = num_hidden_layers

        self._ff = nn.Sequential()
        for i in range(1, self._n_hidden + 1):
            ff_block = nn.Sequential()
            ff_block.add_module(f"FF_Lin", nn.Linear(in_dim, self.Config.FF_HIDDEN_DIM))
            ff_block.add_module(f"FF_BatchNorm", nn.BatchNorm1d(self.Config.FF_HIDDEN_DIM))
            ff_block.add_module(f"FF_Act", self.Config.FF_ACTIVATION())

            self._ff.add_module(f"Hidden_Block_{i}", ff_block)
            in_dim = self.Config.FF_HIDDEN_DIM

        # Add output layer
        if self._n_hidden > 0:
            self._model.add_module("FF", self._ff)
        self._model.add_module("FF_Out", nn.Linear(in_dim, self._n_class))


class PoisonLearner(abc.ABC, nn.Module):
    def __init__(self, n_classes: int, tr_p_dropout: float = 0, te_p_dropout: float = 0):
        super().__init__()
        self._n_classes = n_classes
        self.fc_first = nn.Sequential()
        self.linear = None

        self._tr_p_dropout = self._te_p_dropout = None
        self.tr_p_dropout, self.te_p_dropout = tr_p_dropout, te_p_dropout

    def penultimate(self, x: Tensor) -> Tensor:
        return self.forward(x, penu=True)

    @abc.abstractmethod
    def forward(self, x: Tensor, penu: bool = False, block: bool = False):
        pass

    def build_fc(self, x: Tensor, hidden_dim: Optional[int] = None) -> NoReturn:
        r""" Build the linear block"""
        if hidden_dim is None:
            hidden_dim = 1024

        self.eval()
        with torch.no_grad():
            x = self.forward(x=x, penu=True)

        # Build preclassifier
        kwargs = {"x": x, "in_features": None}
        for block_id in range(1, config.NUM_FF_LAYERS + 1):
            logging.warning("Batch norm disabled in hidden layer")
            block = nn.Sequential(SmartLinear(out_features=hidden_dim, **kwargs),
                                  nn.ReLU(inplace=True))
                                  # nn.ReLU(inplace=True),
                                  # nn.BatchNorm1d(hidden_dim))
            self.fc_first.add_module(f"Hidden Block#{block_id:02}", block)
            kwargs = {"in_features": hidden_dim}

        self.linear = SmartLinear(out_features=self._n_classes, **kwargs)

    def reset_linear(self) -> None:
        r""" Reset the linear layer """
        self.linear.reset_parameters()

    def apply_dropout(self, x: Tensor) -> Tensor:
        r""" Applies a custom dropout layer primarily for improved poison transferability """
        is_train = self.training
        if (is_train and self.tr_p_dropout > 0) or (not is_train and self.te_p_dropout > 0):
            x = F.dropout(x)
        return x

    @property
    def tr_p_dropout(self) -> float:
        r""" Gets the training dropout probability """
        return self._tr_p_dropout

    @tr_p_dropout.setter
    def tr_p_dropout(self, value: float) -> None:
        r""" Sets the training dropout probability """
        assert 0 <= value < 1, "Invalid training dropout probability range"
        self._tr_p_dropout = value

    @property
    def te_p_dropout(self) -> float:
        r""" Gets the test dropout probability """
        return self._te_p_dropout

    @te_p_dropout.setter
    def te_p_dropout(self, value: float) -> None:
        r""" Sets the test dropout probability """
        assert 0 <= value < 1, "Invalid test dropout probability range"
        self._te_p_dropout = value


class SmartLinear(nn.Module):
    def __init__(self, in_features: Optional[int], out_features: int, bias: bool = True,
                 x: Optional[Tensor] = None):
        r"""
        Optionally specifies the input feature count of defines the count on the first pass.
        Also includes a built in \p view invocation
        """
        super().__init__()
        self._view = None

        assert x is None or len(x.shape) == 2, "Bizarre shape of x for SmartLinear layer"

        # If not already created because of unknown feature counts
        if in_features is None:
            if x is None:
                raise ValueError("Must specify either in_features or x")
            in_features = x.shape[1]
        else:
            if x is not None:
                assert in_features == x.shape[1], "Mismatch between in_feature and x's shape"
        self._lin = nn.Linear(in_features, out_features, bias)

    def forward(self, x: Tensor) -> Tensor:
        # If needed, flatten to 1D feature vectors
        if len(x.shape) > 2:
            if self._view is None:
                self._view = ViewTo1D()
            x = self._view.forward(x)

        return self._lin.forward(x)

    def reset_parameters(self) -> None:
        r""" Reset the weights in the linear layer """
        self._lin.reset_parameters()
