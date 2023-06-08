__all__ = [
    "check_success",
    "run",
]

import dill as pk
import logging
from typing import NoReturn, Optional

import torch
from torch import LongTensor, Tensor
import torchvision.transforms as transforms
import wandb

from . import _config as config
from . import dirs
from .influence_utils import InfluenceMethod
from . import learner
from . import gas_and_tracin
from . import tracin_utils
from .types import TensorGroup
from . import utils


def run(block: utils.ClassifierBlock, tg: TensorGroup,
        targ_x: Tensor, targ_y: LongTensor, wd: Optional[float] = None) -> NoReturn:
    r""" Performs end-to-end filtering using epoch information and batch size """
    train_dl, _ = learner.create_fit_dataloader(tg=tg, use_both_classes=True)
    train_dl = tracin_utils.configure_train_dataloader(train_dl)

    tracin_dir = dirs.RES_DIR / "tracin"
    path = utils.construct_filename("inf-res", out_dir=tracin_dir, file_ext="pk")

    # Select the poisoned IDs to remove
    all_inf = gas_and_tracin.calc(block=block, train_dl=train_dl, wd=wd,
                                  n_epoch=config.NUM_EPOCH, bs=config.BATCH_SIZE,
                                  x_targ=targ_x, y_targ=targ_y, ex_id=config.TARG_IDX)

    with open(path, "wb+") as f_out:
        pk.dump(all_inf, f_out)
    gas_and_tracin.log_final(block=block, tensors=all_inf, ex_id=config.TARG_IDX)
    # with open(path, "rb") as f_in:
    #     all_inf = pk.load(f_in)

    fields = (
        (InfluenceMethod.GAS, all_inf.gas_sim),
        (InfluenceMethod.TRACINCP, all_inf.tracincp),
        (InfluenceMethod.TRACIN_RENORM, all_inf.tracin_renorm),
        (InfluenceMethod.TRACIN, all_inf.tracin_inf),
    )
    for method, inf in fields:
        train_dl = tracin_utils.configure_train_dataloader(train_dl=train_dl)
        if config.USE_WANDB:
            tracin_utils.generate_wandb_results(block=block, inf_vals=inf[all_inf.full_ds_ids],
                                                method=method,
                                                ds_ids=all_inf.full_ds_ids,
                                                bd_ids=all_inf.full_bd_ids,
                                                train_dl=train_dl, ex_id=config.TARG_IDX)


def check_success(block: utils.ClassifierBlock,
                  x_targ: Tensor, y_targ: LongTensor, method: Optional[InfluenceMethod] = None,
                  ex_id: Optional[int] = None) -> NoReturn:
    r""" Logs the result of the learner """
    x_targ = config.get_test_tfms()(x_targ).to(utils.TORCH_DEVICE)

    # flds = [block.name()]
    # if method is not None:
    #     flds.append(method.value)
    # flds.append("Poison Cleanse")
    # if ex_id is not None:
    #     flds.append(f"Ex={ex_id}")

    # Compare the prediction to the target
    with torch.no_grad():
        pred = block.module.predict(x=x_targ)

    pred_lbl, targ_lbl = pred.cpu().item(), y_targ.cpu().item()
    is_pois_successful = pred_lbl == targ_lbl
    # flds += ["Target ID:", str(config.TARG_IDX),
    #          "Poison Label:", str(targ_lbl),
    #          "Model Label:", str(pred_lbl),
    #          "Final Result:",
    #          "successful" if is_pois_successful else "FAILED"]

    # logging.info(" ".join(flds))
    assert is_pois_successful, "Poison unnsuccessful"


def _log_target_wandb(targ_x: Tensor, ex_id: int) -> NoReturn:
    r""" Log the target example using W&B """
    if not config.USE_WANDB:
        return
    columns = ["id", "image", "true_label", "adv_label"]
    # Generate the table
    _run = wandb.init(project=utils.get_proj_name())
    res = [ex_id, wandb.Image(transforms.ToPILImage()(targ_x.clamp(0, 1))),
           config.TARG_CLS, config.POIS_CLS]
    inf_table = wandb.Table(data=[res], columns=columns)
    _run.log({f"ex={ex_id}": inf_table})
