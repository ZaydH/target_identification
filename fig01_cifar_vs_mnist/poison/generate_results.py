__all__ = [
    "calculate_results",
]

from dataclasses import dataclass
import io
import logging
import re
import sys
from typing import ClassVar, List, Tuple

import numpy as np
import pycm

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from . import _config as config
from .learner import CombinedLearner
from .types import CustomTensorDataset, TensorGroup
from . import utils


@dataclass(order=True)
class LearnerResults:
    r""" Encapsulates ALL results for a single NLP learner model """
    FIELD_SEP: ClassVar[str] = ","

    @dataclass(init=True, order=True)
    class DatasetResult:
        r""" Encapsulates results of a model on a SINGLE dataset """
        ds_size: int
        accuracy: float = None
        conf_matrix: pycm.ConfusionMatrix = None

    loss_name = None
    valid_loss = None

    inf_func_detect_rate = None
    rep_point_detect_rate = None
    tracin_detect_rate = None

    targ_init_adv_loss = None
    targ_true_loss = None
    targ_adv_loss = None

    tracin_stats = None

    @staticmethod
    def get_general_results_names() -> List[str]:
        r""" Returns the name of each of the results values not tied to a specific dataset """
        return ["valid_loss", "inf_func_detect_rate", "rep_point_detect_rate",
                "tracin_detect_rate", "targ_true_loss", "targ_adv_loss", "targ_init_adv_loss"]


def calculate_results(tg: TensorGroup, erm_learners: CombinedLearner) -> dict:
    r"""
    Calculates and writes to disk the model's results

    :param tg: Tensor group containing the test conditions
    :param erm_learners: Empirical risk minimization based learners
    :return: Dictionary containing results of all experiments
    """
    erm_learners.eval()

    all_res = dict()

    for block_name, block in erm_learners.blocks():  # type: str, utils.ClassifierBlock
        ds_flds = _build_ds_fields(tg=tg, tfms=config.get_test_tfms())

        res = LearnerResults()
        res.loss_name = block_name
        res.valid_loss = block.best_loss
        res.tracin_stats = block.tracin_stats

        for ds_name, ds in ds_flds:
            # noinspection PyTypeChecker
            dl = DataLoader(ds, shuffle=False, drop_last=False, batch_size=config.BATCH_SIZE,
                            num_workers=0)
            all_y, all_yhat = [], []
            with torch.no_grad():
                for xs, ys in dl:
                    all_y.append(ys.cpu())
                    xs = xs.to(torch.device("cuda:0"))
                    all_yhat.append(block.module.predict(xs).cpu())

            # Iterator transforms label so transform it back
            y = torch.cat(all_y, dim=0).view([-1]).numpy()
            y_hat = torch.cat(all_yhat, dim=0).view([-1]).numpy()
            # Store for name "unlabel" or "test"
            single_res = _single_ds_results(block_name, ds_name, y, y_hat)
            res.__setattr__(ds_name, single_res)

        # Append the result
        all_res[block_name] = res

    return all_res


def _single_ds_results(block_name: str,
                       ds_name: str, y: np.ndarray,
                       y_hat: np.ndarray) -> LearnerResults.DatasetResult:
    r""" Logs and returns the results on a single dataset """
    res = LearnerResults.DatasetResult(y.shape[0])

    str_prefix = f"{ds_name}:"

    # Pre-calculate fields needed in other calculations
    res.conf_matrix = pycm.ConfusionMatrix(actual_vector=y, predict_vector=y_hat)

    # noinspection PyUnresolvedReferences
    res.accuracy = res.conf_matrix.Overall_ACC
    logging.debug(f"{str_prefix} Accuracy: {100. * res.accuracy:.3}%")

    # Write confusion matrix to a string
    sys.stdout = cm_out = io.StringIO()
    res.conf_matrix.print_matrix()
    sys.stdout = sys.__stdout__
    # Log the confusion matrix
    cm_str = cm_out.getvalue()
    logging.debug(f"{str_prefix} Confusion Matrix: \n{cm_str}")
    res.conf_matrix_str = re.sub(r"\s+", " ", str(cm_str.replace("\n", " ")))

    return res


def _build_ds_fields(tg: TensorGroup, tfms):
    r""" Builds dataset fields.  Poisoned data my not be included if it has not been created yet """
    return (
            ("Target Instance", CustomTensorDataset([tg.targ_x, tg.targ_y], tfms)),
            ("Test (Clean)", CustomTensorDataset([tg.te_cl_x, tg.te_cl_y], tfms)),
            ("Test (Adversarial)", CustomTensorDataset([tg.te_bd_x, tg.te_bd_y], tfms)),
           )


def _calc_predict_vector(block: utils.ClassifierBlock, x: Tensor, tfms) -> Tensor:
    r""" Construct a prediction vector for the \p create backdoor success heat map """
    ds = CustomTensorDataset([x], tfms)
    dl = DataLoader(ds, batch_size=config.BATCH_SIZE, drop_last=False, shuffle=False,
                    num_workers=0)

    with torch.no_grad():
        y_hat = [block.module.predict(xs) for xs, in dl]
    y_hat = torch.cat(y_hat, dim=0)
    return y_hat


def _extract_targ_pois_idx(conf_matrix: pycm.ConfusionMatrix) -> Tuple[int, int]:
    r"""
    Extracts the index of the target and poison classes respectively from the confusion matrix's
    classes.  If the class number does not appear in the class list, -1 is returned.
    """
    classes = conf_matrix.classes
    idx_lst = []
    for cls_id in (config.TARG_CLS, config.POIS_CLS):
        for i, val in enumerate(classes):
            try:
                val = int(val)
            except ValueError:  # Cannot convert val to string
                continue
            if int(val) == cls_id:
                idx_lst.append(i)
                break
        else:
            idx_lst.append(-1)
    return tuple(idx_lst)  # noqa
