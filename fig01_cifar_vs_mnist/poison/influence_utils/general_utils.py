__all__ = [
    "ADV_LABEL",
    "CLEAN_LABEL",
    "InfluenceMethod",
    "MIN_LOSS",
    "build_log_start_flds",
    "calc_adv_auprc",
    "calc_adv_auroc",
    "calc_cutoff_detect_rates",
    "calc_identified_adv_frac",
    "calc_pearsonr",
    "calc_spearmanr",
    "check_bd_ids_contents",
    "check_duplicate_ds_ids",
    "count_adv",
    "derivative_of_loss",
    "is_bd",
    "label_ids",
    "log_time",
]

import enum
import logging
import time
from typing import Callable, NoReturn, Optional, Sequence

import scipy.stats
import sklearn.metrics

import torch
from torch import BoolTensor, LongTensor, Tensor

from .. import _config as config

ADV_LABEL = +1
CLEAN_LABEL = -1
MIN_LOSS = 1E-12


class InfluenceMethod(enum.Enum):
    r""" Influence method of interest """
    INF_FUNC = "Influence Function"
    INF_FUNC_RENORM = f"{INF_FUNC} Renormalized"

    REP_POINT = "Representer Point"
    REP_POINT_RENORM = f"{REP_POINT} Renormalized"

    TRACINCP = "TracInCP"

    GAS = "GAS"
    GAS_L = f"{GAS}-Layerwise"

    LOSS_ADV_SPOT = "Adv. Loss"
    LOSS_CLEAN_SPOT = "Clean Loss"

    GRAD_NORM_MAG_RATIO = "Gradient Norm Magnitude Ratio"

    TRACIN = "TracIn"
    TRACIN_RENORM = f"{TRACIN} Renormalized"


def check_duplicate_ds_ids(ds_ids: Tensor) -> NoReturn:
    r""" Ensure there are no duplicate dataset IDs """
    assert ds_ids.dtype == torch.long, "Dataset IDs does not appear to be longs"
    uniq = torch.unique(ds_ids)
    assert uniq.shape[0] == ds_ids.shape[0], "Duplicate dataset IDs should not occur"


def check_bd_ids_contents(bd_ids: Tensor) -> NoReturn:
    r""" Ensure not too many backdoor IDs """
    assert bd_ids.dtype == torch.long, "Adversarial IDs does not appear to be longs"
    uniq = torch.unique(bd_ids)
    # Add 1 since one possible additional value for non-backdoor examples
    assert uniq.shape[0] <= config.OTHER_CNT + 1


def calc_identified_adv_frac(block, res_type: InfluenceMethod, helpful_bd_ids: Tensor,
                             ep: Optional[int] = None, subepoch: Optional[int] = None,
                             ex_id: Optional[int] = None) -> float:
    r""" Calculates and logs the number of examples from poisoned blocks """
    assert helpful_bd_ids.dtype == torch.long, "helpful_ids does not appear to be a list of IDs"
    check_bd_ids_contents(bd_ids=helpful_bd_ids)

    # Extract the most influential samples:
    n_ex = config.OTHER_CNT
    labels = label_ids(bd_ids=helpful_bd_ids[:n_ex])
    bd_only = labels[labels == ADV_LABEL]

    # Extract the subset of IDs that are poison
    frac_pois = bd_only.shape[0] / n_ex

    flds_val = build_log_start_flds(block=block, res_type=res_type,
                                    ep=ep, subepoch=subepoch, ex_id=ex_id)
    logging.info(f"{flds_val} Adv. Detected: {frac_pois:.1%}")
    return frac_pois


def build_log_start_flds(block, res_type: Optional[InfluenceMethod],
                         ep: Optional[int] = None, subepoch: Optional[int] = None,
                         ex_id: Optional[int] = None) -> str:
    r""" Creates the log starter fields """
    # flds = [block.name()]
    flds = []
    # if ex_id is not None:
    #     flds.append(f"Ex={ex_id}")
    flds.append(_construct_ep_str(ep=ep, subepoch=subepoch))
    if res_type is not None:
        flds.append(res_type.value)
    return " ".join(flds)


def calc_cutoff_detect_rates(block, res_type: InfluenceMethod, helpful_bd_ids: Tensor,
                             ep: Optional[int] = None, subepoch: Optional[int] = None,
                             ex_id: Optional[int] = None) -> NoReturn:
    r"""
    Logs how much poison is detected at various cutoffs

    :param block: Block under investigation
    :param res_type:
    :param helpful_bd_ids: List of backdoor IDs ordered from most helpful to least helpful
    :param ep: Epoch number
    :param subepoch: Optional subepoch number
    :param ex_id: Optional example ID number
    """
    flds_str = build_log_start_flds(block=block, res_type=res_type,
                                    ep=ep, subepoch=subepoch, ex_id=ex_id)

    # is_bd = label_ids(helpful_bd_ids) == BACKDOOR_LABEL
    bd_mask = is_bd(helpful_bd_ids)
    n_bd = torch.sum(bd_mask).item()  # number of backdoor
    for percent in (0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 10, 20):
        count = int(percent / 100. * helpful_bd_ids.numel())
        p_bd = torch.sum(bd_mask[:count]).item()  # Number of backdoor at specified percent
        denom = min(n_bd, count)
        rate = p_bd / denom  # Fraction backdoor detected

        msg = f"{flds_str} Backdoor {percent}% Data: {p_bd} / {denom} ({rate:.1%})"
        logging.info(msg)


def label_ids(bd_ids: Tensor, n_bd: Optional[int] = None) -> Tensor:
    r"""
    Poison is treated as the positive class with label "+1".  Clean data is treated as the negative
    class with label "-1".
    :param bd_ids: List of training set IDs.  Not required to be ordered.
    :param n_bd: number of backdoor examples
    :return: One-to-one mapping of the training set IDs to either the poison or clean classes
    """
    check_bd_ids_contents(bd_ids=bd_ids)

    # assert n_bd > 0, "At least one backdoor example is expected"
    if n_bd is None or n_bd == 0:
        n_bd = config.OTHER_CNT

    labels = torch.full(bd_ids.shape, fill_value=CLEAN_LABEL, dtype=torch.long)
    mask = bd_ids < n_bd

    labels[mask] = ADV_LABEL
    return labels


def is_bd(bd_ids: Tensor, n_bd: Optional[int] = None) -> BoolTensor:
    r""" Returns whether each ID is a backdoor """
    lbls = label_ids(bd_ids=bd_ids, n_bd=n_bd)
    return lbls == ADV_LABEL


def count_adv(bd_ids: Tensor) -> int:
    r""" Counts the number of poison examples """
    return torch.sum(is_bd(bd_ids)).item()


def calc_adv_auprc(block, res_type: InfluenceMethod,
                   bd_ids: Tensor, ds_ids: Tensor, inf: Tensor,
                   ep: Optional[int] = None, subepoch: Optional[int] = None,
                   ex_id: Optional[int] = None) -> float:
    r""" Calculate the block's AUPRC """
    return _base_roc_calc(is_auroc=False, block=block, res_type=res_type,
                          bd_ids=bd_ids, ds_ids=ds_ids, inf=inf,
                          ep=ep, subepoch=subepoch, ex_id=ex_id)


def calc_adv_auroc(block, res_type: InfluenceMethod,
                   bd_ids: Tensor, ds_ids: Tensor, inf: Tensor,
                   ep: Optional[int] = None, subepoch: Optional[int] = None,
                   ex_id: Optional[int] = None) -> float:
    r""" Calculate the block's AUROC """
    return _base_roc_calc(is_auroc=True, block=block, res_type=res_type,
                          bd_ids=bd_ids, ds_ids=ds_ids, inf=inf,
                          ep=ep, subepoch=subepoch, ex_id=ex_id)


def _base_roc_calc(is_auroc: bool, block, res_type: InfluenceMethod,
                   bd_ids: Tensor, ds_ids: Tensor, inf: Tensor,
                   ep: Optional[int] = None, subepoch: Optional[int] = None,
                   ex_id: Optional[int] = None) -> float:
    r"""
    Calculate and log the ROC

    :param is_auroc: If \p True, return the AUROC
    :param block: Block of interest
    :param res_type: Result type to be stored
    :param bd_ids: Backdoor IDs used to determine number of backdoor samples
    :param ds_ids: Training example IDs
    :param inf: Corresponding influence values for the list of training ids
    :param ep: If specified, AUROC is reported for a specific epoch
    :param subepoch: If specified, subepoch value
    :return: AUC value
    """
    check_duplicate_ds_ids(ds_ids=ds_ids)
    check_bd_ids_contents(bd_ids=bd_ids)
    assert bd_ids.shape == ds_ids.shape == inf.shape, "IDs and influence values do not match"

    labels = label_ids(bd_ids=bd_ids)

    if is_auroc:
        # noinspection PyUnresolvedReferences
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, inf)
        # noinspection PyUnresolvedReferences
        roc_val = sklearn.metrics.auc(fpr, tpr)
    else:
        # noinspection PyUnresolvedReferences
        prec, recall, _ = sklearn.metrics.precision_recall_curve(y_true=labels, probas_pred=inf)
        # noinspection PyUnresolvedReferences
        roc_val = sklearn.metrics.average_precision_score(labels, inf)

    roc_name = "AUROC" if is_auroc else "AUPRC"
    # flds = [block.name()]
    flds = []
    # if ex_id is not None:
    #     flds.append(f"Ex={ex_id}")
    flds += [_construct_ep_str(ep=ep, subepoch=subepoch),
             res_type.value, f"{roc_name}:", f"{roc_val:.3f}"]
    msg = " ".join(flds)
    logging.info(msg)

    return roc_val


def log_time(res_type: InfluenceMethod):
    r""" Logs the running time of each influence method """
    def decorator(func):
        r""" Need to nest the decorator since decorator takes an argument (\p res_type) """
        def wrapper(*args, **kwargs) -> NoReturn:
            start = time.time()

            rets = func(*args, **kwargs)

            total = time.time() - start
            logging.info(f"{res_type.value} Execution Time: {total:,.6f} seconds")

            return rets
        return wrapper
    return decorator


def _construct_ep_str(ep: Optional[int], subepoch: Optional[int]) -> str:
    r""" Helper method to standardize constructing the epoch strings """
    if ep is None:
        assert subepoch is None, "Subepoch is specified without an epoch"
        ep_str = "Final"
    else:
        ep_str = f"Ep {ep}"
        if subepoch is not None:
            ep_str = f"{ep_str}.{subepoch:03}"
    return ep_str


def flatten(vec: Sequence[Tensor]) -> Tensor:
    r""" Flatten the gradient into a vector """
    return torch.cat([flat.detach().view([-1]) for flat in vec], dim=0)


# Use global variable to prevent reinitializing memory
gbl_layer_norm = None


def build_layer_norm(grad) -> Tensor:
    r""" Construct a layerwise norm vector """
    assert len(grad) > 1, "Flatten vector is not supported"

    global gbl_layer_norm
    if gbl_layer_norm is None:
        gbl_layer_norm = [vec.clone().detach() for vec in grad]

    assert len(gbl_layer_norm) == len(grad), "Unexpected length mismatch"
    for layer, vec in zip(gbl_layer_norm, grad):  # type: Tensor, Tensor
        norm = vec.detach().norm().item()
        if norm == 0:
            norm = 1E-8
        layer.fill_(norm)
    return flatten(gbl_layer_norm)


def _error_check_correlation_tensors(x: Tensor, y: Tensor) -> NoReturn:
    r""" Standardizes checks for correlation variables """
    assert x.numel() == y.numel(), "Mismatch in number of elements"
    assert len(x.shape) <= 2 and x.shape[0] == x.numel(), "x tensor has a bizarre shape"
    assert len(y.shape) <= 2 and y.shape[0] == y.numel(), "y tensor has a bizarre shape"


def calc_pearsonr(x: Tensor, y: Tensor) -> float:
    r""" Calculates Pearson coefficient between the \p x and \p y tensors """
    _error_check_correlation_tensors(x=x, y=y)
    x, y = x.numpy(), y.numpy()
    r, _ = scipy.stats.pearsonr(x=x, y=y)
    return r


def calc_spearmanr(x: Tensor, y: Tensor) -> float:
    r""" Calculates Pearson coefficient between the \p x and \p y tensors """
    _error_check_correlation_tensors(x=x, y=y)
    x, y = x.numpy(), y.numpy()
    r, _ = scipy.stats.spearmanr(a=x, b=y)
    return r


def derivative_of_loss(acts: Tensor, lbls: LongTensor, f_loss: Callable) -> Tensor:
    r"""
    Calculates the dervice of loss function \p f_loss w.r.t. output activation \p outs
    and labels \p lbls
    """
    for tensor, name in ((acts, "acts"), (lbls, "lbls")):
        assert len(tensor.shape) <= 2 and tensor.numel() == 1, f"Unexpected shape for {name}"
    # Need to require gradient to calculate derive
    acts = acts.detach().clone()
    acts.requires_grad = True
    # Calculates the loss
    loss = f_loss(acts, lbls).view([1])
    ones = torch.ones(acts.shape[:1], dtype=acts.dtype, device=acts.device)
    # Back propagate the gradients
    loss.backward(ones)
    return acts.grad.clone().detach().type(acts.dtype)  # type: Tensor
