__all__ = [
    "calc",
]

import copy
import json
import logging
from pathlib import Path
import pickle as pk
from typing import NoReturn, Optional, Tuple

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from . import _config as config
from . import dirs
from . import influence_utils
from .influence_utils import InfluenceMethod
from .influence_utils.nn_influence_utils import InfFuncTensors
from .losses import TORCH_DEVICE
from . import tracin_utils
from .types import CustomTensorDataset, LearnerParams
from . import utils

DEFAULT_R = 10
INF_RES_DIR = None


@influence_utils.log_time(res_type=InfluenceMethod.INF_FUNC)
def calc(block: utils.ClassifierBlock, tr_dl: DataLoader, te_x: Tensor,
         te_y: Tensor, ex_id: Optional[int] = None, use_precompute: bool = False):
    r"""
    :param block: Block of interest
    :param tr_dl: \p DataLoader used to train the learners
    :param te_x:
    :param te_y:
    :param ex_id:
    :param use_precompute:
    :return:
    """
    global INF_RES_DIR
    INF_RES_DIR = dirs.RES_DIR / config.DATASET.name.lower() / "inf_func" / block.name().lower()
    INF_RES_DIR.mkdir(exist_ok=True, parents=True)

    # Use the tensors to create a supervised training set. Supports optional test transform
    kw_args = {}
    if config.has_tfms():
        kw_args = {"transform": config.get_test_tfms()}
    ds = CustomTensorDataset([te_x, te_y], **kw_args)
    te_dl = DataLoader(ds, drop_last=False, shuffle=False, num_workers=utils.NUM_WORKERS,
                       batch_size=1)

    # Do not use the original training dataloader since it may require transforms or drop part of
    # the dataset.  Create temporary Dataloader that overcomes those issues
    assert isinstance(tr_dl.dataset, CustomTensorDataset), "Code supports certain dataset type"

    # Each learner will have different influence function values
    _calc_block_inf(block=block, tr_dl=tr_dl, te_dl=te_dl, ex_id=ex_id,
                    use_precompute=use_precompute)


def _calc_block_inf(block: utils.ClassifierBlock, tr_dl: DataLoader, te_dl: DataLoader,
                    ex_id: Optional[int],
                    use_precompute: bool) -> InfFuncTensors:
    r"""
    Encapsulates calculating the influence of blocks

    :param block: Block for which influence is calculated
    :param tr_dl: Train \p DataLoader
    :param te_dl: Test \p DataLoader
    :return: Tuple of the sorted estimated loss, backdoor IDs, and dataset IDs
    """
    block.eval()
    wd = config.get_learner_val(block.name(), LearnerParams.Attribute.WEIGHT_DECAY)

    bl_x, bl_y, bd_ids, ds_ids = _build_learner_tensors(block=block, train_dl=tr_dl)
    n_tr = bl_y.shape[0]
    batch_tr_dl, instance_tr_dl = _build_block_dataloaders(bl_x=bl_x, bl_y=bl_y)

    flds = [block.name(), "inf-fin"]
    prefix = "-".join(flds).lower()
    # noinspection PyTypeChecker
    filename = utils.construct_filename(prefix=prefix, out_dir=INF_RES_DIR, ex_id=ex_id,
                                        file_ext="pkl", add_ds_to_path=False)
    # if not filename.exists():
    msg = f"calculation of {InfluenceMethod.INF_FUNC.value} influence"
    logging.info(f"Starting {msg}")
    if True:
        precomputed_s_test = None
        if use_precompute:
            assert filename.exists(), "Precomputed s_test file does not exist"
            with open(filename, "rb") as f_in:
                precomputed_s_test = pk.load(f_in).s_test

        # Order of res: influences, train_inputs_collections, s_test
        res = influence_utils.compute_influences(model=block,
                                                 n_gpu=1 if torch.cuda.is_available() else 0,
                                                 device=TORCH_DEVICE,
                                                 f_loss=block.loss.calc_train_loss,
                                                 test_dl=te_dl,
                                                 batch_train_data_loader=batch_tr_dl,
                                                 instance_train_data_loader=instance_tr_dl,
                                                 weight_decay=wd,
                                                 s_test_damp=config.DAMP,
                                                 s_test_scale=config.SCALE,
                                                 s_test_num_samples=config.R_DEPTH,
                                                 s_test_iterations=config.T_REPEATS,
                                                 precomputed_s_test=precomputed_s_test)

        # Extract the result fields
        res.ds_ids, res.bd_ids = ds_ids.clone(), bd_ids.clone()
        with open(filename, "wb+") as f_out:
            pk.dump(res, f_out)
    logging.info(f"COMPLETED {msg}")

    with open(filename, "rb") as f_in:
        res = pk.load(f_in)  # type: InfFuncTensors
    flds = (
        (res.inf_base, InfluenceMethod.INF_FUNC),
        (res.inf_sim, InfluenceMethod.INF_FUNC_RENORM),
    )
    for influence_vals, method in flds:
        # Convert the estimated loss
        est_loss = -1 / n_tr * influence_vals

        est_loss_sorted, helpful = torch.sort(est_loss, dim=0, descending=True)
        tmp_bd_ids, tmp_ds_ids = res.bd_ids[helpful], res.ds_ids[helpful]

        tracin_utils.results.generate_epoch_stats(ep=None, subepoch=None, method=method,
                                                  block=block, inf_vals=est_loss_sorted,
                                                  bd_ids=tmp_bd_ids, ds_ids=tmp_ds_ids,
                                                  ex_id=ex_id, log_cutoff=True)

        if config.USE_WANDB:
            train_dl = tracin_utils.configure_train_dataloader(train_dl=tr_dl)
            tracin_utils.generate_wandb_results(block=block, inf_vals=est_loss_sorted.cpu(),
                                                ds_ids=tmp_ds_ids.cpu(), bd_ids=tmp_bd_ids.cpu(),
                                                method=method, train_dl=train_dl, ex_id=ex_id)

    return res


def _log_influence_results(block: utils.ClassifierBlock, est_loss: Tensor, helpful: Tensor,
                           ds_ids: Tensor, bl_y: Tensor) -> NoReturn:
    r"""
    :param block:
    :param est_loss: Estimated change in loss if training example is removed
    :param helpful: Training examples numbered from 0 to (# training examples - 1)
    :param ds_ids: Dataset ID numbers for the training examples used by the block
    :param bl_y: Labels for the training examples used by the block
    """
    b_name = block.name()
    influence_utils.check_duplicate_ds_ids(ds_ids=ds_ids)

    harmful = torch.flip(helpful, dims=[0])
    for i in range(2):
        if i == 0:
            name = "helpful"
        else:
            name = "harmful"
        top_ord = locals()[name][:5]
        top_ids = ds_ids[top_ord]
        logging.info(f"{b_name}: Top {name} IDs: {top_ids.tolist()}")
        logging.info(f"{b_name}: Top {name} Est. Change Loss: {est_loss[top_ord].tolist()}")
        logging.info(f"{b_name}: Top {name} Labels: {bl_y[top_ord].tolist()}")


def _build_inf_results_file(block: utils.ClassifierBlock, helpful: Tensor, influence_vals: Tensor,
                            est_loss_vals: Tensor, ds_ids: Tensor, bl_y: Tensor) -> NoReturn:
    r"""
    Constructs the influence results file

    :param block:
    :param helpful:
    :param influence_vals:
    :param est_loss_vals:
    :param ds_ids: Dataset ID numbers for the training examples used by the block
    :param bl_y: Labels for the training examples used by the block
    :return:
    """
    assert ds_ids.shape == helpful.shape, "Helpful tensor shape does not match the ID tensor"
    assert bl_y.shape == helpful.shape, "Helpful tensor shape does not match the y tensor"

    ord_ids, ord_y = ds_ids[helpful], bl_y[helpful]
    influence_utils.check_duplicate_ds_ids(ds_ids=ds_ids)

    ord_inf, ord_est_loss = influence_vals[helpful], est_loss_vals[helpful]
    inf_res = {"block_name": block.name(),
               "dataset": config.DATASET.name,
               "hvp_batch_size": config.HVP_BATCH_SIZE,
               "damp": config.DAMP,
               "scale": config.SCALE,
               "helpful-ids": ord_ids.tolist(),
               "helpful-ord-loss": ord_est_loss.tolist(),
               "helpful-y": ord_y.tolist(),
               "helpful-influence": ord_inf.tolist(),
               "test_id": config.TARG_IDX,
               "test-targ-cls": config.TARG_CLS,
               "test-pois-cls": config.POIS_CLS
               }

    res_path = _build_inf_res_filename(block)
    with open(res_path, "w+") as f_out:
        json.dump(inf_res, f_out)


def _build_inf_res_filename(block: utils.ClassifierBlock) -> Path:
    r""" Construct the filename for the results """
    prefix = f"inf-{block.name()}-t-id={config.TARG_IDX}"
    return utils.construct_filename(prefix, out_dir=dirs.RES_DIR, file_ext="json",
                                    add_timestamp=True)


def _build_learner_tensors(block: utils.ClassifierBlock,
                           train_dl: DataLoader) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    Construct the X/y/IDs tensors based on any example filtering by the block
    :param block:
    :param train_dl:
    :return: Tuple of X, y, backdoor IDs, and dataset IDs tensors respectively
    """
    tmp_ds = copy.copy(train_dl.dataset)
    tmp_ds.transform = None
    cp_tr_dl = DataLoader(tmp_ds, batch_size=config.BATCH_SIZE,
                          drop_last=False, shuffle=False, num_workers=utils.NUM_WORKERS)

    verify_contents = True   # ToDo Revert to false to speed up final version

    all_x, all_y, all_bd_ids, all_ds_ids = [], [], [], []
    for batch_tensors in cp_tr_dl:
        batch = block.organize_batch(batch_tensors, process_mask=True,
                                     verify_contents=verify_contents)
        if batch.skip():
            continue

        all_x.append(batch.xs.cpu())
        all_y.append(batch.lbls.cpu())
        all_bd_ids.append(batch.bd_ids.cpu())
        all_ds_ids.append(batch.ds_ids.cpu())

    all_ds_ids, id_ordering = torch.sort(torch.cat(all_ds_ids, dim=0), dim=0)
    # Consolidate into tensors and sort by the image IDs
    n_tr = id_ordering.shape[0]  # Number of training examples
    # Strange count for number of training examples
    assert n_tr == id_ordering.numel(), "Weird size mismatch"

    filename = "_".join([block.name().lower(), "inf-func", block.start_time])
    # noinspection PyUnresolvedReferences
    np.savetxt(INF_RES_DIR / (filename + "_ord-id.csv"), all_ds_ids.numpy(), fmt='%d',
               delimiter=',')

    # Combine the subtensors and ensure the order aligns with all_ds_ids
    tr_x, tr_y = torch.cat(all_x, dim=0)[id_ordering], torch.cat(all_y, dim=0)[id_ordering]
    all_bd_ids = torch.cat(all_bd_ids, dim=0)[id_ordering]
    return tr_x, tr_y, all_bd_ids, all_ds_ids


def _build_block_dataloaders(bl_x: Tensor, bl_y: Tensor) -> Tuple[DataLoader, DataLoader]:
    r"""
    Constructs two separate dataloaders.  You may want different properties dataloader properties
    when estimating the Hessian vector product (HVP) and when estimating influence.  By specifying
    separate \p DataLoaders, those two roles are separated.

    :param bl_x: Block's X tensor
    :param bl_y: Blocks y (i.e. label) tensor
    :return: Tuple of the batch \p DataLoader (used for generating the HVP) and the instance
             \p DataLoader used when estimating influence.
    """
    # ToDo Determine whether to use transforms
    ds = CustomTensorDataset((bl_x, bl_y), transform=config.get_train_tfms())
    batch_tr_dl = DataLoader(ds, batch_size=config.HVP_BATCH_SIZE,
                             shuffle=True, drop_last=True, num_workers=utils.NUM_WORKERS)

    ds = CustomTensorDataset((bl_x, bl_y), transform=config.get_test_tfms())
    instance_tr_dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False,
                                num_workers=utils.NUM_WORKERS)
    return batch_tr_dl, instance_tr_dl
