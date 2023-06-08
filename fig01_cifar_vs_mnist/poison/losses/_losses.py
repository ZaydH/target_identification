__all__ = [
            "Loss",
            "RiskEstimatorBase",
            "TORCH_DEVICE",
            "ce_loss",
          ]

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn

from .. import _config as config

TORCH_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# _log_ce_module = nn.CrossEntropyLoss(reduction="none")
_log_ce_module = None


def ce_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    r""" Cross-entropy loss that takes two arguments instead of default one """
    # clamp_val = 1e7
    global _log_ce_module
    if _log_ce_module is None:
        if config.DATASET.is_mnist() or config.DATASET.is_cifar():
            _log_ce_module = nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise ValueError("Unknown dataset.  Cannot set cross entropy loss weights.")

    if config.DATASET.is_mnist() or config.DATASET.is_cifar():
        targets = targets.float() if inputs.dtype == torch.float else targets.double()
    return _log_ce_module.forward(inputs, targets)
    # return _log_ce_module.forward(inputs.clamp(-clamp_val, clamp_val), targets)


class RiskEstimatorBase(ABC):
    def __init__(self, n_bd: int, train_loss: Callable, valid_loss: Optional[Callable] = None,
                 name_suffix: str = ""):
        if valid_loss is not None:
            valid_loss = train_loss

        self.tr_loss = train_loss
        self.val_loss = valid_loss

        self.n_bd = n_bd

        self._name_suffix = name_suffix

    @abstractmethod
    def name(self) -> str:
        r""" Name of the risk estimator """

    def is_poisoned(self) -> bool:
        r""" Returns \p True if the estimator uses backdoored data """
        return self.n_bd > 0

    def calc_train_loss(self, dec_scores: Tensor, labels: Tensor, **kwargs) -> Tensor:
        r""" Calculates the risk using the TRAINING specific loss function """
        return self._loss(dec_scores=dec_scores, lbls=labels, f_loss=self.tr_loss, **kwargs)

    def calc_validation_loss(self, dec_scores: Tensor, labels: Tensor, **kwargs) -> Tensor:
        r""" Calculates the risk using the VALIDATION specific loss function """
        return self._loss(dec_scores=dec_scores, lbls=labels, f_loss=self.val_loss, **kwargs)

    def build_mask(self, ids: Tensor) -> Tensor:
        r""" Construct the examples to which the backdoor will be applied """
        mask = ids < self.n_bd  # type: Tensor
        return mask

    @staticmethod
    def has_any(mask: Tensor) -> bool:
        r""" Checks if the mask has any set to \p True """
        assert mask.dtype == torch.bool, "Mask should be a Boolean Tensor"
        return bool(mask.any().item())

    @abstractmethod
    def _loss(self, dec_scores: Tensor, lbls: Tensor, f_loss: Callable, **kwargs) -> Tensor:
        r""" Single function for calculating the loss """


class Loss(RiskEstimatorBase):
    RETURN_MEAN_KEY = "return_mean"

    def name(self) -> str:
        flds = [f"" if self.is_poisoned() else "L-clean"]
        if self._name_suffix:
            flds.append(self._name_suffix)
        return "".join(flds)

    def _loss(self, dec_scores: Tensor, lbls: Tensor, f_loss: Callable, **kwargs) -> Tensor:
        r""" Straight forward PN loss -- No weighting by prior & label """
        is_binary = config.DATASET.is_cifar() or config.DATASET.is_mnist()
        assert is_binary or len(dec_scores.shape) == 2, "Bizarre input shape"
        assert dec_scores.shape[0] == lbls.shape[0], "Vector shape loss mismatch"

        lst_loss = f_loss(dec_scores, lbls)
        if self.RETURN_MEAN_KEY in kwargs and not kwargs[self.RETURN_MEAN_KEY]:
            return lst_loss
        return lst_loss.mean()
