__all__ = [
    "process_epoch",
]

import logging
import sys
from typing import NoReturn, Optional

import torch
import tqdm
from torch import LongTensor, Tensor

from . import _settings as settings
from . import results
from . import utils
from .. import _config as config
from .. import influence_utils
from ..influence_utils import InfluenceMethod
from .. import types as parent_types
from .. import utils as parent_utils


def process_epoch(block32: parent_utils.ClassifierBlock,
                  dataset: parent_types.CustomTensorDataset,
                  id_map: dict, ep: int, ep_wd: float, ep_bs: int,
                  x_targ32: Tensor, y_targ: LongTensor,
                  tensors: utils.TracInTensors, ex_id: Optional[int]) -> NoReturn:
    r"""
    Performs TracIn on a single epoch (including subepochs if applicable) for the specified
    \p block

    :param block32: Block for use with floats (i.e., float32)
    :param dataset: Dataset object of interest
    :param id_map: Maps example ID to dataset index
    :param ep: Active epoch number
    :param ep_wd: Epoch weight decay value
    :param ep_bs: Epoch batch_size value
    :param x_targ32: Already transformed X-target
    :param y_targ: y of target example
    :param tensors: All results tensors
    :param ex_id: Optional target example ID number
    :return:
    """
    assert isinstance(dataset, parent_types.CustomTensorDataset), "Dataset class is not supported"
    influence_utils.check_duplicate_ds_ids(ds_ids=tensors.full_ds_ids)
    influence_utils.check_bd_ids_contents(bd_ids=tensors.full_bd_ids)

    def _load_blocks(_ep: int, _subep: Optional[int]):
        r""" Standardizes loading the block parameters """
        block32.restore_epoch_params(ep=_ep, subepoch=_subep)
        block32.eval()

    cur_subep = 0
    _load_blocks(_ep=ep - 1, _subep=None)
    # Continue learning rate from end of the last epoch
    lr = block32.get_prev_lr_val(ep=ep - 1, subepoch=None)

    # Epoch dataset IDs ordered by batches
    ep_ds_ids = block32.get_epoch_dataset_ids(ep=ep)
    subep_ends = block32.get_subepoch_ends(ep=ep)
    n = len(id_map)

    # Iterate through the subepochs
    n_subep = len(block32.get_subepoch_ends(ep=ep))
    tqdm_desc = f"Epoch {ep} Subepoch %d"
    for cur_subep in range(n_subep + 1):
        # Get subepoch IDs for TracIn
        start_rng = subep_ends[cur_subep - 1] if cur_subep > 0 else 0
        end_rng = subep_ends[cur_subep] if cur_subep < len(subep_ends) - 1 else n
        subep_ids = ep_ds_ids[start_rng:end_rng]

        # Subepoch used to load stored data
        subep_load = cur_subep if cur_subep < n_subep else None

        # Initialize the tensors storing the results from the subepoch
        tensors.subep.reset()

        # Get the loss gradient for the test (target) example
        loss_targ32, acts_targ32, grad_targ32 = utils.compute_grad(block32, ep_wd,
                                                                   x_targ32, y_targ,
                                                                   flatten=True)

        always_use_dbl = utils.is_grad_zero(grad=grad_targ32)
        if always_use_dbl:
            header = influence_utils.build_log_start_flds(block=block32, ep=ep, subepoch=cur_subep,
                                                          res_type=None, ex_id=ex_id)

        # Skip iter if target has zero gradient even at double precision
        skip_iter = always_use_dbl and utils.is_grad_zero(grad=grad_targ32)

        ex_desc = f"Epoch {ep} Subepoch: {cur_subep}"
        ex_tqdm = tqdm.tqdm(tensors.full_ds_ids, total=tensors.full_ds_ids.shape[0],
                            file=sys.stdout, disable=config.QUIET, desc=ex_desc)
        with ex_tqdm as ex_bar:
            if not skip_iter:  # skip if targ grad is zero
                for cnt, id_val in enumerate(ex_bar):
                    utils.tracin_dot_product(block32=block32,
                                             grad_targ32=grad_targ32,
                                             subep_tensors=tensors.subep,
                                             ds=dataset, id_val=id_val, id_map=id_map,
                                             ep_wd=ep_wd)
            else:
                loss_targ32.fill_(settings.MIN_LOSS)
                acts_targ32.fill_(settings.MIN_LOSS)
                # Prevent divide by zero errors in later calculations
                tensors.subep.grad_norms.fill_(settings.MIN_NORM)
                tensors.subep.loss_vals.fill_(settings.MIN_LOSS)

        # Perform normalization based on learning rate and batch size as specified by TracIn
        # Perform externally to make faster with CUDA
        tensors.subep.dot_vals *= lr / ep_bs
        # Perform the sqrt externally to use CUDA
        tensors.subep.grad_norms.sqrt_()

        _combine_and_log_results(block=block32, ep=ep, subep=cur_subep,
                                 grad_targ=grad_targ32, subep_ids=subep_ids,
                                 tensors=tensors, ex_id=ex_id)

        # Load parameters and learning rate for the next (sub)epoch
        _load_blocks(_ep=ep, _subep=subep_load)
        lr = block32.get_prev_lr_val(ep=ep, subepoch=subep_load)


def _combine_and_log_results(block: parent_utils.ClassifierBlock, ep: int, subep: int,
                             subep_ids: LongTensor,
                             grad_targ: Tensor,
                             tensors: utils.TracInTensors,
                             ex_id: Optional[int]) -> NoReturn:
    r""" Combines and logs all results """
    full_ds_ids, full_bd_ids = tensors.full_ds_ids, tensors.full_bd_ids
    tensors.subep.dot_normed = tensors.subep.dot_vals / tensors.subep.grad_norms

    # Equivalent of TracInCp
    tensors.tracincp[full_ds_ids] += tensors.subep.dot_vals[full_ds_ids]

    # GAS Results
    targ_grad_norm = grad_targ.norm()
    targ_grad_norm[targ_grad_norm <= 0] = settings.MIN_NORM
    gas_sim_base = tensors.subep.dot_normed / targ_grad_norm.cpu()
    tensors.gas_sim[full_ds_ids] += gas_sim_base[full_ds_ids]

    # TracIn Results
    tensors.tracin_inf[subep_ids] += tensors.subep.dot_vals[subep_ids]
    # TracIn normalized by L2 cosine norm
    tensors.tracin_renorm[subep_ids] += gas_sim_base[subep_ids]


def _log_ratio_stats(block: parent_utils.ClassifierBlock, ep: int, subep: int,
                     vals: Tensor, full_ds_ids: LongTensor,
                     full_bd_ids: LongTensor, ex_id: Optional[int],
                     is_grad_norm: bool) -> NoReturn:
    r""" Calculates and returns the adversarial and clean mean norms respectively """
    assert full_bd_ids.numel() == full_ds_ids.numel(), "Adversarial/dataset length mismatch"
    # Extract only the relevant cumulative IDs
    assert vals.numel() > torch.max(full_ds_ids).item(), "Some dataset ID not found"
    vals = vals[full_ds_ids]

    # Label whether each example is a backdoor or not
    is_bd = influence_utils.is_bd(bd_ids=full_bd_ids)

    adv_vals, cl_vals = vals[is_bd], vals[~is_bd]
    if not is_grad_norm:
        res_types = (InfluenceMethod.LOSS_CLEAN_SPOT, InfluenceMethod.LOSS_ADV_SPOT)
        for vals, r_type in zip((cl_vals, adv_vals), res_types):
            utils.log_vals_stats(block=block, ep=ep, subep=subep, res_type=r_type, norms=vals,
                                 ex_id=ex_id)
    else:
        ratio_res_type = InfluenceMethod.GRAD_NORM_MAG_RATIO
        # Log mean and median ratios for clear documentation
        header = influence_utils.build_log_start_flds(block=block, ep=ep, subepoch=subep,
                                                      res_type=ratio_res_type, ex_id=ex_id)
        # mean_mag_ratio = (adv_vals.mean() / cl_vals.mean()).view([1])
        # logging.info(f"{header} Mean: {mean_mag_ratio.item():.3E}")

        median_mag_ratio = (adv_vals.median() / cl_vals.median()).view([1])
        logging.info(f"{header} Median: {median_mag_ratio.item():.3E}")


def _log_vals_split_adv_clean(block: parent_utils.ClassifierBlock, ep: int, subep: int,
                              clean_method: InfluenceMethod, adv_method: InfluenceMethod,
                              vals: Tensor, full_ds_ids: LongTensor, full_bd_ids: LongTensor,
                              ex_id: Optional[int]) -> NoReturn:
    is_clean = ~influence_utils.is_bd(bd_ids=full_bd_ids)
    utils.log_vals_stats(block=block, res_type=clean_method, ep=ep, subep=subep,
                         norms=vals[full_ds_ids[is_clean]], ex_id=ex_id)
    # Log Adversarial stats
    is_bd = influence_utils.is_bd(bd_ids=full_bd_ids)
    utils.log_vals_stats(block=block, res_type=adv_method, ep=ep, subep=subep,
                         norms=vals[full_ds_ids[is_bd]], ex_id=ex_id)
