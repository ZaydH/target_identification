__all__ = [
    "calc_representer_vals"
]

import logging
import time
from typing import NoReturn, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from . import _config as config
from . import dirs
from . import influence_utils
from .influence_utils import InfluenceMethod
from . import utils
from .learner import CombinedLearner
from . import tracin_utils
from .types import LearnerParams
from .utils import ClassifierBlock, TORCH_DEVICE


def calc_representer_vals(erm_learners: CombinedLearner, train_dl: DataLoader,
                          test_x: Tensor, method: InfluenceMethod = InfluenceMethod.REP_POINT,
                          ex_id: Optional[int] = None) -> dict:
    r"""

    :param erm_learners:
    :param train_dl:
    :param test_x:
    :param method: Version of representer point method to use
    :param ex_id: Optional target example ID number
    :return: Dictionary containing the representer values for each test example in \p test_x.
             Dimension of the representer tensor is #Test x #Classes x #Examples.
    """
    start = time.time()

    ret = _calc_representer_vals(erm_learners=erm_learners, train_dl=train_dl,
                                 test_x=test_x, method=method, ex_id=ex_id)

    total = time.time() - start
    logging.info(f"{method.value} Execution Time: {total:,.6f} seconds")
    return ret


def _calc_representer_vals(erm_learners: CombinedLearner, train_dl: DataLoader,
                           test_x: Tensor, method: InfluenceMethod,
                           ex_id: Optional[int] = None) -> dict:
    r"""

    :param erm_learners:
    :param train_dl:
    :param test_x:
    :param method: Influence method being calculated
    :param ex_id: Optional target example ID number
    :return: Dictionary containing the representer values for each test example in \p test_x.
             Dimension of the representer tensor is #Test x #Classes x #Examples.
    """
    # Transform the test tensor
    test_x = config.get_test_tfms()(test_x).to(utils.TORCH_DEVICE)
    # Modify the dataloader to not drop last and to disable augmentation
    train_dl = tracin_utils.configure_train_dataloader(train_dl=train_dl)

    # msg = f"Calculating representer values"
    # if ex_id is not None:
    #     msg += f" for Ex={ex_id}"
    # logging.info(f"Starting: {msg}")

    _calc_alpha_i(erm_learners, train_dl, method=method)

    res = dict()
    for block_name, block in erm_learners.blocks():
        block.eval()

        # noinspection PyPep8Naming
        w_dot_T = _calc_feature_dot_product(train_dl=train_dl, test_x=test_x, block=block)

        # w_dot is dimension  #Test x 1 x #TrainExamples.  Feature dot product f_i^T f_t is
        # independent of the class label.
        w_dot = torch.transpose(w_dot_T, 0, 1).unsqueeze(dim=1)

        # Matrix for \alpha_i in the paper
        alpha_i_raw = block.alpha_i
        # Initial dimension of alpha_i is 1 x #Classes x #TrainExamples since the alpha_i values
        # are independent of the test examples.
        alpha_i = torch.transpose(alpha_i_raw, 0, 1).unsqueeze(dim=0)

        elewise_pro = w_dot.to(TORCH_DEVICE) * alpha_i.to(TORCH_DEVICE)
        elewise_pro = elewise_pro.cpu()

        res[block_name] = (block, elewise_pro)

    # logging.info(f"COMPLETED: {msg}")
    return res


def _calc_alpha_i(erm_learners: CombinedLearner, train_dl: DataLoader,
                  method: InfluenceMethod) -> NoReturn:
    r"""
    Calculates the set of :math:`\alpha_i` values used in representer point calculations

    :param erm_learners: Set of trained learners
    :param train_dl: DataLoader used in training
    :param method: Influence method being calculated
    """
    # Training set elements EXCLUDING validation set. config.N_TRAIN includes validation set
    n_tr_only = sum([batch_tensors[0].shape[0] for batch_tensors in train_dl])

    for block_name, block in erm_learners.blocks():  # type: str, utils.ClassifierBlock
        block.eval()
        pois_module = block.get_module()

        # Initialize the alpha values
        alpha_i = torch.zeros([config.N_FULL_TR, config.N_CLASSES], dtype=tracin_utils.DTYPE)
        all_ds_ids, all_bd_ids = [], []

        # Get parameter values
        f_loss = block.loss.calc_train_loss  # Loss function used for calculating gradient
        lmbd = config.get_learner_val(block.name(), LearnerParams.Attribute.WEIGHT_DECAY)
        alpha_scale = -1. / (2 * lmbd * n_tr_only)

        for batch_tensors in train_dl:
            batch = block.organize_batch(batch_tensors, process_mask=True)
            if batch.skip():
                continue

            with torch.no_grad():
                xs = pois_module.forward(batch.xs, penu=True)

            with torch.no_grad():
                dec_scores = pois_module.linear.forward(xs)
            if config.DATASET.is_mnist() or config.DATASET.is_cifar():
                # Ensure proper shape for BCE loss
                dec_scores = dec_scores.squeeze(dim=1)
            # Use higher-bit data type for influence estimation to reduce likelihood of underflow
            # for normalization
            dec_scores = dec_scores.type(tracin_utils.DTYPE)
            dec_grad = influence_utils.derivative_of_loss(acts=dec_scores, lbls=batch.lbls,
                                                          f_loss=f_loss)
            dec_grad *= alpha_scale

            if method in (InfluenceMethod.REP_POINT_RENORM,):
                dec_grad.sign_()

            # Populate the alpha list
            alpha_i[batch.ds_ids] = dec_grad.cpu()
            all_bd_ids.append(batch.bd_ids)
            all_ds_ids.append(batch.ds_ids)

        # Store the alpha IDs and alpha_i values
        block.alpha_ds_ids, sort_indices = torch.sort(torch.cat(all_ds_ids, dim=0))
        block.alpha_bd_ids = torch.cat(all_bd_ids, dim=0)[sort_indices]
        block.alpha_i = alpha_i[block.alpha_ds_ids]


def _calc_feature_dot_product(train_dl: DataLoader, block: utils.ClassifierBlock,
                              test_x: Tensor) -> Tensor:
    r"""
    As detailed in representer theorem 3.1, representer point calculation requires calculating
    :math:`f_i^T f_t`.

    :param train_dl: \p DataLoader containing the samples used during training
    :param block: Function used to encode the feature vectors, i.e., transform :math:`x` into
                  :math:`f`.
    :param test_x: Test vectors to be considered

    :return: Weight dot product.  Dimension is [# training & poison samples vs. # test samples]
    """
    n_test = test_x.shape[0]

    block.eval()

    with torch.no_grad():
        test_x = block.get_module().forward(test_x.to(TORCH_DEVICE), penu=True)
    test_x_T = torch.transpose(test_x, 0, 1).to(TORCH_DEVICE)  # noqa

    w_dot = torch.zeros([config.N_FULL_TR, n_test], dtype=torch.float)
    for batch_tensors in train_dl:
        batch = block.organize_batch(batch_tensors, process_mask=True)
        if batch.skip():
            continue

        with torch.no_grad():
            dec_out = block.get_module().forward(batch.xs, penu=True)

        dot_prod = dec_out @ test_x_T
        w_dot[batch.ds_ids] = dot_prod.cpu()

    w_dot = w_dot[block.alpha_ds_ids]
    return w_dot


def _error_check_grad(n_rows: int, dec_scores: Tensor, lbls: Tensor) -> Tensor:
    r""" Calculate manually the gradient """
    exp_vec = torch.exp(dec_scores)
    denom = exp_vec.sum(dim=1).unsqueeze(dim=1)

    offset = torch.zeros_like(dec_scores, device=TORCH_DEVICE, dtype=tracin_utils.DTYPE)
    for row in range(0, n_rows):
        offset[row][lbls[row]] = -1

    # offset_np = offset.numpy()  # For checking in the debugger

    return offset + exp_vec / denom


def log_representer_scores(idx: int, lbl: int, rep_dict: dict, file_prefix: str,
                           ex_id: Optional[int], method: InfluenceMethod,
                           train_dl: DataLoader) -> NoReturn:
    # Construct the representer file name
    log_dir = dirs.RES_DIR / config.DATASET.name.lower() / "rep_vals"
    log_dir.mkdir(exist_ok=True, parents=True)

    for block_name, (block, rep_vals) in rep_dict.items():  # type: str, (ClassifierBlock, Tensor)
        flds = ["rep-vals"]

        if file_prefix:
            flds.append(file_prefix)
        flds += [block_name.lower().replace("_", "-"), f"lbl={lbl}", f"idx={config.TARG_IDX}"]
        path = utils.construct_filename(prefix="_".join(flds), out_dir=log_dir, file_ext="csv",
                                        add_ds_to_path=False, add_timestamp=True)
        if path.exists():
            continue

        rep_vals = rep_vals[idx, lbl]  # type: Tensor
        argsort = torch.argsort(rep_vals, dim=0, descending=True)

        # Sort representer in decreasing order and
        bd_ids, ds_ids = block.alpha_bd_ids.cpu(), block.alpha_ds_ids.cpu()
        assert bd_ids.shape[0] == ds_ids.shape[0] == rep_vals.shape[0] == argsort.shape[0]
        ds_ids, bd_ids, rep_vals = ds_ids[argsort], bd_ids[argsort], rep_vals[argsort]

        tracin_utils.results.generate_epoch_stats(ep=None, subepoch=None, method=method,
                                                  block=block, inf_vals=rep_vals,
                                                  bd_ids=bd_ids, ds_ids=ds_ids, ex_id=ex_id,
                                                  log_cutoff=True)

        # Print a formatted file of representer point values
        with open(str(path), "w+") as f_out:
            f_out.write(f"helpful-ids,{block.name()}\n")
            for i in range(0, ds_ids.shape[0]):
                f_out.write(f"{ds_ids[i].item()},{rep_vals[i].item():.6E}\n")

        if config.USE_WANDB:
            train_dl = tracin_utils.configure_train_dataloader(train_dl=train_dl)
            tracin_utils.generate_wandb_results(block=block, inf_vals=rep_vals,
                                                method=method, ds_ids=ds_ids, bd_ids=bd_ids,
                                                train_dl=train_dl, ex_id=ex_id)
