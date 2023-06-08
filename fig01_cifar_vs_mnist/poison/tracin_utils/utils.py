__all__ = [
    "DTYPE",
    "TracInTensors",
    "compute_grad",
    "configure_train_dataloader",
    "export_tracin_epoch_inf",
    "flatten_grad",
    "generate_wandb_results",
    "get_gas_log_flds",
    "get_tracincp_log_flds",
    "get_topk_indices",
    "get_tracin_log_flds",
    "log_vals_stats",
    "sort_ids_and_inf",
    "tracin_dot_product",
]

import copy
import dataclasses
import logging
import pickle as pk
from typing import List, NoReturn, Optional, Tuple, Union

import torch
from torch import BoolTensor, DoubleTensor, FloatTensor, LongTensor, Tensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb

from . import _settings as settings
from .. import _config as config
from .. import dirs
from .. import influence_utils
from ..influence_utils import InfluenceMethod
from ..influence_utils import nn_influence_utils
from .. import utils as parent_utils

DTYPE = torch.double


@dataclasses.dataclass
class SubepochTensors:
    # Dot product values
    dot_vals: Tensor
    # Loss values
    loss_vals: Tensor
    # Gradient norms
    grad_norms: Tensor
    # Dot producted normalized by norm
    dot_normed: Tensor

    def __init__(self, inf_numel: int):
        # Initialize all the fields to the same tensor shape
        for f in dataclasses.fields(self):
            if f.type == BoolTensor:
                dtype = torch.bool
            elif f.type == LongTensor:
                dtype = torch.long
            elif f.type == Tensor:
                dtype = DTYPE
            else:
                raise ValueError("Unknown type to copy")
            tensor = torch.zeros([inf_numel], dtype=dtype, requires_grad=False)
            setattr(self, f.name, tensor)

    def reset(self) -> NoReturn:
        r""" Reset the tensor at the start of an epoch """
        for f in dataclasses.fields(self):
            if f.type == Tensor:
                clone = torch.zeros_like(self.__getattribute__(f.name))
                setattr(self, f.name, clone)
        self.dot_normed = None  # noqa


@dataclasses.dataclass
class TracInTensors:
    # full_ds_ids: Full set of dataset IDs used
    full_ds_ids: LongTensor
    # full_bd_ids: Full set of backdoor IDs used
    full_bd_ids: LongTensor
    # grain_unnorm: Modified in place tensor simulating GrAIn without gradient normalization
    tracincp: Tensor
    # gas_sim: Modified in place tensor storing the GAS values
    gas_sim: Tensor
    # tracin_inf: Modified in place tensor storing the TracIn values
    tracin_inf: Tensor
    tracin_renorm: Tensor
    # List of magnitude ratios throughout training
    magnitude_ratio: List[Tensor]
    # Subepoch Tensors
    subep: SubepochTensors

    def __init__(self, full_ds_ids: LongTensor, full_bd_ids: LongTensor, inf_numel: int):
        inf_base = torch.zeros([inf_numel], dtype=DTYPE, requires_grad=False)
        # Initialize all the fields to the same tensor shape
        for f in dataclasses.fields(self):
            if f.type == Tensor:
                setattr(self, f.name, inf_base.clone())

        # Stores the adv/clean magnitude ratio
        self.magnitude_ratio = []
        # Store the IDs
        self.full_ds_ids = full_ds_ids
        self.full_bd_ids = full_bd_ids
        # Store the number of zero gradients for each element
        self.adv_zeros = torch.zeros([inf_numel], dtype=torch.long,  # type: LongTensor # noqa
                                     requires_grad=False)
        self.tot_zeros = self.adv_zeros.clone()
        # Subepoch tensors
        self.subep = SubepochTensors(inf_numel=inf_numel)


def configure_train_dataloader(train_dl: DataLoader) -> DataLoader:
    r"""" Configure the DataLoader for use in TracIn """
    # Switch to the test transform and update the train dataloader to not drop points/shuffle
    ds = copy.copy(train_dl.dataset)
    ds.set_transform(config.get_test_tfms())  # noqa
    # Cannot use the new method since torch does not let you change the dataset of an initialized
    # dataloader
    new_tr_dl = DataLoader(ds, batch_size=1, num_workers=train_dl.num_workers,
                           drop_last=False, shuffle=False)
    new_tr_dl.tfm = config.get_train_tfms()
    return new_tr_dl


def sort_ids_and_inf(inf_arr: Tensor, bd_ids_arr: Tensor,
                     ds_ids_arr: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    r""" Helper method for sorting the IDs and in """
    assert inf_arr.dtype == DTYPE, "Influence array is not the correct datatype"
    influence_utils.check_bd_ids_contents(bd_ids=bd_ids_arr)
    influence_utils.check_duplicate_ds_ids(ds_ids=ds_ids_arr)
    assert bd_ids_arr.shape[0] == ds_ids_arr.shape[0] == inf_arr.shape[0], "Num ele mismatch"

    ord_ids = torch.argsort(inf_arr, dim=0, descending=True)

    ord_inf = inf_arr.clone()[ord_ids]
    ord_bd_ids, ord_ds_ids = bd_ids_arr.clone()[ord_ids], ds_ids_arr.clone()[ord_ids]
    return ord_inf, ord_bd_ids, ord_ds_ids


def export_tracin_epoch_inf(all_in: bool, block: parent_utils.ClassifierBlock,
                            ep_inf: List[Tuple[Tensor, Tensor, Tensor]]) -> NoReturn:
    r""" Backup-up the TracIn data for later post-processing """
    outdir = dirs.RES_DIR / config.DATASET.name.lower() / "tracin" / block.name().lower() / "ep-inf"
    outdir.mkdir(parents=True, exist_ok=True)

    desc = "all-in" if all_in else "sep"
    path = parent_utils.construct_filename(prefix=f"ep-inf-{desc}", out_dir=outdir, file_ext="pk",
                                           add_ds_to_path=False, add_timestamp=True)
    with open(str(path), "wb+") as f_out:
        pk.dump(ep_inf, f_out)


def compute_grad(block, ep_wd, _x: Tensor, _y: Tensor, is_dbl: bool = False,
                 flatten: bool = True) -> Tuple[Tensor, Tensor, Union[FloatTensor, DoubleTensor]]:
    r"""
    Helper method to standardize gradient computation
    :return: Tuple of the loss, output activations, and gradient respectively
    """
    assert _x.shape[0] == 1 and _y.shape[0] == 1, "Only single example supported"
    if is_dbl:
        _x = _x.double()
        # Need to cast to double when using BCE loss
        if config.DATASET.is_mnist() or config.DATASET.is_cifar():
            _y = _y.double()
    loss, acts, grad = nn_influence_utils.compute_gradients(device=parent_utils.TORCH_DEVICE,
                                                            model=block, n_gpu=0,
                                                            f_loss=block.loss.calc_train_loss,
                                                            x=_x, y=_y, weight_decay=ep_wd,
                                                            params_filter=None,
                                                            weight_decay_ignores=None,
                                                            create_graph=False,
                                                            return_loss=True, return_acts=True)
    if flatten:
        grad = flatten_grad(grad).detach()
    else:
        grad = [vec.detach() for vec in grad]
    return loss.detach(), acts.detach(), grad  # noqa


def tracin_dot_product(block32: parent_utils.ClassifierBlock,
                       grad_targ32: FloatTensor,
                       id_val: Tensor,
                       subep_tensors: SubepochTensors,
                       ds, id_map: dict, ep_wd: Optional[float]) -> NoReturn:
    r"""
    Computes the TracIn dot product

    :param block32: Block for use with floats (i.e., float32)
    :param id_val:
    :param grad_targ32: 32-bit version of target gradient vector
    :param ds:
    :param id_map:
    :param ep_wd: Weight decay for the epoch (if applicable)
    :param subep_tensors: Subepoch tensors
    """
    batch_tensors = ds[[id_map[id_val.item()]]]
    batch = block32.organize_batch(batch_tensors, process_mask=True)
    assert len(batch) == 1, "Only singleton batches supported"

    def _calc_prods() -> Tuple[Tensor, Tensor]:
        return grad_dot_prod(grad_x, grad_targ), grad_dot_prod(grad_x, grad_x)

    loss, acts, grad_x = compute_grad(block32, ep_wd, batch.xs, batch.lbls, flatten=True)
    grad_targ = grad_targ32
    fin_dot, dot_prod = _calc_prods()

    if dot_prod <= 0:
        dot_prod = settings.MIN_NORM

    if loss.item() <= 0:  # noqa
        loss.fill_(settings.MIN_LOSS)

    lderiv = influence_utils.derivative_of_loss(f_loss=block32.loss.calc_train_loss,
                                                acts=acts, lbls=batch.lbls.to(acts.device))  # noqa
    lderiv.abs_()
    if lderiv.item() == 0:
        lderiv.fill_(settings.MIN_LOSS)

    # Add the gradient dot product
    subep_tensors.dot_vals[id_val] = fin_dot
    subep_tensors.grad_norms[id_val] = dot_prod
    subep_tensors.loss_vals[id_val] = loss


def grad_dot_prod(grad_1: Tensor, grad_2: Tensor) -> Tensor:
    r""" Gradient dot product """
    # assert grad_1.shape == grad_2.shape, "Shape mismatch"
    # assert prod.shape == grad_1.shape, "Weird shape after product"
    return torch.dot(grad_1, grad_2)


def flatten_grad(grad: Tuple[Union[FloatTensor, DoubleTensor], ...]) \
        -> Union[FloatTensor, DoubleTensor]:
    r""" Flattens gradients into a single continguous vector """
    return torch.cat([vec.view([-1]) for vec in grad if vec is not None], dim=0)  # noqa


def get_topk_indices(grad: Tensor, frac: float) -> Tensor:
    assert 0 < frac < 1, "Fraction of indices to keep"
    k = int(frac * grad.numel())
    _, idx = torch.topk(grad.abs(), k=k)
    mask = torch.zeros_like(grad, dtype=torch.bool)
    mask[idx] = True
    return mask


def generate_wandb_results(block: parent_utils.ClassifierBlock,
                           method: influence_utils.InfluenceMethod,
                           inf_vals: Tensor, ds_ids: LongTensor, bd_ids: LongTensor,
                           train_dl: DataLoader,
                           ex_id: Optional[int] = None) -> NoReturn:
    r"""
    Generate a summary of the results to using W&B

    :param block: Block being analyzed
    :param method: Influence estimation method
    :param inf_vals: Influence values
    :param ds_ids:
    :param bd_ids:
    :param train_dl:
    :param ex_id:
    """
    if not config.USE_WANDB:
        return
    # max_ds_id = torch.max(ds_ids).item()
    # assert max_ds_id < inf_vals.numel(), "Dataset IDs do not exist for all elements"
    assert ds_ids.numel() == inf_vals.numel() == bd_ids.numel(), "Length mismatch"

    logging.debug(f"Generating W&B {method.value} influence results table")

    # Sort all the tensors to be safe
    influence_utils.check_bd_ids_contents(bd_ids=bd_ids)
    influence_utils.check_duplicate_ds_ids(ds_ids=ds_ids)
    inf_vals, _, ds_ids = sort_ids_and_inf(inf_arr=inf_vals, bd_ids_arr=bd_ids,
                                           ds_ids_arr=ds_ids)

    # Get rank of each training example
    ramp = torch.arange(1, ds_ids.numel() + 1, dtype=torch.long)
    ds_rank = torch.full([torch.max(ds_ids) + 1], fill_value=-1, dtype=torch.long)
    ds_inf = torch.zeros_like(ds_rank, dtype=DTYPE)
    ds_rank[ds_ids], ds_inf[ds_ids] = ramp, inf_vals

    # create a wandb.Table() with corresponding columns
    columns = ["id", "image", "inf", "rank", "label", "is_bd"]
    #  Construct images of all results
    to_pil = transforms.ToPILImage()
    all_res = []
    train_dl = configure_train_dataloader(train_dl=train_dl)
    for batch_tensors in train_dl:
        batch = block.organize_batch(batch_tensors, process_mask=True)
        if batch.skip():
            continue

        id_val = batch.ds_ids.item()  # Id number of the example
        is_bd = influence_utils.label_ids(bd_ids=batch.bd_ids) == influence_utils.ADV_LABEL
        # Construct the results
        tr_ex = [id_val, wandb.Image(to_pil(batch.xs[0].clamp(0, 1))),
                 ds_inf[id_val].item(), ds_rank[id_val].item(),
                 batch.lbls.item(), is_bd.item()]
        all_res.append(tr_ex)

    # Generate the table
    run = wandb.init(project=parent_utils.get_proj_name())
    inf_table = wandb.Table(data=all_res, columns=columns)
    flds = [method.value, "inf-sum"]
    run.log({"_".join(flds).lower().replace(" ", "-"): inf_table})


def log_vals_stats(block: parent_utils.ClassifierBlock, res_type: InfluenceMethod,
                   ep: Optional[int], subep: Optional[int], norms: Tensor,
                   ex_id: Optional[int]) -> NoReturn:
    r""" Standardizing method for logging norm mean and standard deviation """
    header = influence_utils.build_log_start_flds(block=block, ep=ep, subepoch=subep,
                                                  res_type=res_type, ex_id=ex_id)

    # Calculate quantiles
    quantiles = torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE)
    names = ["25%-Quartile", "Median", "75%-Quartile"]
    quant_vals = torch.quantile(norms, q=quantiles)
    for name, val in zip(names, quant_vals.tolist()):
        logging.info(f"{header} {name}: {val:.2E}")
    # # Interquartile range
    # val = quant_vals[-2] - quant_vals[1]
    # logging.info(f"{header} IQR: {val.item():.6E}")

    # std, mean = torch.std_mean(norms, unbiased=True)
    # for val, val_name in zip((mean, std), ("Mean", "Stdev")):
    #     logging.info(f"{header} {val_name}: {val.item():.6E}")


def is_grad_zero(grad: Tensor) -> bool:
    r""" Returns \p True if the gradient dot product is zero """
    dot_prod = grad_dot_prod(grad, grad)
    return dot_prod.item() == 0


layer_norm32 = None

def get_tracincp_log_flds(tensors: TracInTensors) \
        -> Tuple[Tuple[Tensor, InfluenceMethod, str], ...]:
    r""" Construct the GrAIn-based TracIn result fields for logging """
    return (
        (tensors.tracincp, InfluenceMethod.TRACINCP, "grain-unnorm"),
    )


def get_gas_log_flds(tensors: TracInTensors) -> Tuple[Tuple[Tensor, InfluenceMethod, str], ...]:
    r""" Construct the GAS result fields for logging """
    return (
        (tensors.gas_sim, InfluenceMethod.GAS, "gas"),
    )


def get_tracin_log_flds(tensors: TracInTensors) -> Tuple[Tuple[Tensor, InfluenceMethod, str], ...]:
    r""" Construct the TracIn result fields for logging """
    return (
        (tensors.tracin_inf, InfluenceMethod.TRACIN, "tracin"),
        (tensors.tracin_renorm, InfluenceMethod.TRACIN_RENORM, "tracin-sim"),
    )
