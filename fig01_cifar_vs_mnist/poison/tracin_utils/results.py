__all__ = [
    "export",
    "generate_epoch_stats",
]

from pathlib import Path
from typing import NoReturn, Optional, Tuple, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from . import utils
from .. import _config as config
from .. import dirs
from .. import influence_utils
from ..influence_utils import InfluenceMethod
from .. import utils as parent_utils


def export(block: parent_utils.ClassifierBlock, epoch: Optional[int], subepoch: Optional[int],
           bd_ids: Tensor, ds_ids: Tensor, vals: Tensor, res_desc: str,
           sort_inf: bool = True, ex_id: Optional[int] = None) -> NoReturn:
    r"""
    Export the TracIn results files

    :param block: Classifier block used
    :param epoch: Results epoch number.  If not specified, then results are treated as final
    :param subepoch: Subepoch number.  If not specified treated as epoch's final results.
    :param bd_ids: ID denoting whether example is backdoor or not
    :param ds_ids: Dataset IDs used by the block
    :param vals: Influence scores of the training examples
    :param res_desc: Unique descriptor included in the filename
    :param sort_inf: If \p True, sort the influences before printing them
    :param ex_id: Optional target example ID number
    """
    influence_utils.check_bd_ids_contents(bd_ids=bd_ids)
    influence_utils.check_duplicate_ds_ids(ds_ids=ds_ids)
    assert ds_ids.shape[0] == bd_ids.shape[0] == vals.shape[0], "TracIn tensor shape mismatch"

    inf_dir = _build_tracin_res_dir(block=block)

    if epoch is not None:
        inf_dir /= f"ep{epoch:03d}_{block.start_time}"
    inf_dir.mkdir(exist_ok=True, parents=True)

    # Add the epoch information to the filename.  Optionally include the subepoch naming
    if epoch is None:
        ep_desc = "fin"
    else:
        assert epoch < 10 ** 4, "Invalid epoch count as cause filenames out of order"
        ep_desc = f"ep={epoch:03d}"
        if subepoch is not None:
            assert subepoch < 10 ** 3, "Invalid subepoch cnt as would cause filenames out of order"
            ep_desc = f"{ep_desc}.{subepoch:03d}"
    # Construct full file prefix
    flds = [block.name().lower(), "tracin", ep_desc, res_desc]
    file_prefix = "_".join(flds)

    # Add time only if epoch is None since when epoch is not None, the time stamp is added to
    # the folder path.  See above.
    filename = parent_utils.construct_filename(prefix=file_prefix, out_dir=inf_dir, file_ext="csv",
                                               add_ds_to_path=False, add_timestamp=epoch is None,
                                               ex_id=ex_id)

    if sort_inf:
        vals, bd_ids, ds_ids = utils.sort_ids_and_inf(inf_arr=vals, bd_ids_arr=bd_ids,
                                                      ds_ids_arr=ds_ids)
    # Write the TracIn influence to a file. Specify those examples that are actually backdoored
    is_bd = influence_utils.label_ids(bd_ids, n_bd=block.num_adv)
    with open(str(filename), "w+") as f_out:
        f_out.write("ds_ids,vals,is_adv\n")
        for i in range(0, vals.shape[0]):
            f_out.write(f"{ds_ids[i].item()},{vals[i].item():.8E},{is_bd[i].item()}\n")


def _build_tracin_res_dir(block: parent_utils.ClassifierBlock) -> Path:
    r""" Constructs the default directory where TracIn results are written """
    inf_dir = dirs.RES_DIR / config.DATASET.name.lower() / "tracin" / block.name().lower()
    inf_dir.mkdir(exist_ok=True, parents=True)
    return inf_dir


def _store_stats_in_block(block: parent_utils.ClassifierBlock) -> NoReturn:
    r"""
    Records and stores the TracIn result statistics with the block so they can be included
    in the final results
    """
    tracin_inf = block.tracin_fin_inf

    # Stats combining all points
    tracin_stats = block.tracin_stats
    tracin_stats.total_inf = torch.sum(tracin_inf).item()
    tracin_stats.abs_inf = torch.sum(tracin_inf.abs()).item()

    tracin_stats.topk_inf = torch.sum(tracin_inf[:config.OTHER_CNT]).item()
    tracin_stats.botk_inf = torch.sum(tracin_inf[-config.OTHER_CNT:]).item()


def generate_epoch_stats(ep: Optional[int], subepoch: Optional[int], method: InfluenceMethod,
                         block: parent_utils.ClassifierBlock,
                         inf_vals: Tensor, ds_ids: Tensor, bd_ids: Tensor,
                         ex_id: int, log_cutoff: bool = False) -> NoReturn:
    r""" Generate the epoch PDR, AUROC, AUPRC, etc """
    assert inf_vals.shape == bd_ids.shape == ds_ids.shape, "Mismatch in results shape"

    kwargs = {"block": block, "ex_id": ex_id, "res_type": method,
              "ep": ep, "subepoch": subepoch}

    # Calculate and store the epoch's AUPRC
    # path = _build_auc_path_name(block=block, ep=ep, subepoch=subepoch, res_name="auprc")
    influence_utils.calc_adv_auprc(bd_ids=bd_ids, ds_ids=ds_ids, inf=inf_vals, **kwargs)


def export_combined(block, all_in: bool,
                    full_ds_ids: Tensor, full_bd_ids: Tensor, valid_dl: DataLoader,
                    ep_inf_info: List[Tuple[str, Tensor]]) -> NoReturn:
    r"""
    Exports the combined TracIn influences

    :param block: Block under analysis
    :param all_in: Whether the results correspond to all-in or subepoch
    :param full_ds_ids: Full set of dataset IDs
    :param full_bd_ids: Full set of backdoor IDs
    :param valid_dl: Validation dataset
    :param ep_inf_info:
    """
    influence_utils.check_bd_ids_contents(bd_ids=full_bd_ids)
    influence_utils.check_duplicate_ds_ids(ds_ids=full_ds_ids)

    # Split the descriptions and the influences
    all_desc = [desc for desc, _ in ep_inf_info]
    ep_inf = [inf_val.view([-1, 1]) for _, inf_val in ep_inf_info]
    # Each row is a example and each column is a different granularity
    combined_ep_info = torch.cat(ep_inf, dim=1)[full_ds_ids]

    bd_labels = influence_utils.label_ids(bd_ids=full_bd_ids)
    is_valid = label_valid(block=block, full_ds_ids=full_ds_ids, valid_dl=valid_dl)

    assert bd_labels.numel() == full_ds_ids.numel() == is_valid.numel(), "Num elements mismatch"

    out_dir = dirs.RES_DIR / config.DATASET.name.lower() / "tracin-full-log" / block.name().lower()
    for file_cnt in range(2):
        # Construct the filename depending on the data being written
        flds = ["individ-inf" if file_cnt == 0 else "cum-inf", block.name().lower(),
                "all-in" if all_in else "subep"]
        filename = parent_utils.construct_filename(prefix="_".join(flds), out_dir=out_dir,
                                                   file_ext="csv", add_ds_to_path=False,
                                                   add_timestamp=True)
        filename.parent.mkdir(exist_ok=True, parents=True)

        with open(str(filename), "w+") as f_out:
            # Write the file header
            f_out.write("id,is_adv,is_valid")
            for desc in all_desc:
                f_out.write(f",{desc}")
            f_out.write("\n")

            for row in range(full_ds_ids.numel()):
                # Writes each ID number and it is poison or not
                line = [f"{full_ds_ids[row].item()}", f"{bd_labels[row].item()}",
                        f"{is_valid[row].item()}"]
                f_out.write(",".join(line))

                # Writes the influence value
                for col in range(combined_ep_info.shape[1]):
                    f_out.write(f",{combined_ep_info[row, col].item():.10E}")
                f_out.write("\n")

        # Calculate cumulative influence for second pass
        if file_cnt == 0:
            combined_ep_info = torch.cumsum(combined_ep_info, dim=1)


def label_valid(block, full_ds_ids: Tensor, valid_dl: DataLoader) -> Tensor:
    r""" Label the elements in the validation sense """
    is_valid = torch.full((full_ds_ids.max().item() + 1,), fill_value=-1, dtype=torch.long)
    valid_dl = utils.configure_train_dataloader(train_dl=valid_dl)
    for batch_tensors in valid_dl:
        batch = block.organize_batch(batch_tensors, process_mask=True)
        is_valid[batch.ds_ids] = 1
    return is_valid[full_ds_ids]
