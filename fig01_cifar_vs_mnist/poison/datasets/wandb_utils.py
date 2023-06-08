__all__ = [
    "upload_data",
]

import logging
import os
from pathlib import Path
from typing import List, NoReturn, Optional

import PIL  # noqa

from torch import Tensor
import torchvision
import wandb

from .. import _config as config
from ..types import TensorGroup
from .. import utils as parent_utils

# logging.getLogger('wandb').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

RAW_PREFIX = "raw"
SPLIT_PREFIX = "split"


def upload_data(tg: TensorGroup, labels: Optional[List[str]]) -> NoReturn:
    if not config.USE_WANDB:
        return

    _create_raw_images(tg=tg, labels=labels)

    _split_data()


def _create_raw_images(tg: TensorGroup, labels: Optional[List[str]]) -> NoReturn:
    r""" Upload perturbation data to W&B """
    # Top level directory storing the data
    raw_dir = _build_proj_dir(RAW_PREFIX)

    # if this is a substantially new dataset, give it a new name
    # this will create a whole new placeholder (Artifact) for this dataset
    # instead of just incrementing a version of the old dataset
    run = wandb.init(project=parent_utils.get_proj_name())  # , job_type="upload")
    # create an artifact for all the raw data
    # create an artifact for all the raw data
    raw_data_at = wandb.Artifact(str(raw_dir), type="raw_data")

    # config.print_configuration()
    parent_utils.log_seeds()

    split_counts = dict()
    for prefix in ("tr", "bd", "val", "targ", "te_cl", "te_bd"):
        x = tg.__getattribute__(f"{prefix}_x").clone()
        y = tg.__getattribute__(f"{prefix}_y").clone()
        ids = tg.__getattribute__(f"{prefix}_ids").clone()

        cur_dir = raw_dir / prefix
        cur_dir.mkdir(exist_ok=True, parents=True)
        split_counts[prefix] = y.numel()

        for idx in range(ids.numel()):
            # Construct the image filename
            _y_val = y[idx].item()
            y_lbl_suffix = "" if labels is None else labels[_y_val]

            _img_dir = cur_dir / f"{_y_val}_{y_lbl_suffix}"
            _img_dir.mkdir(exist_ok=True, parents=True)
            name = f"{ids[idx].item():06d}.png"
            path = _img_dir / name
            # Write the image file
            _save_image(x[idx], path=path)
            # add file to artifact by full path
            raw_data_at.add_file(str(path), name=str(_img_dir / name))

    # save artifact to W&B
    run.log_artifact(raw_data_at)
    run.finish()


def _save_image(tensor: Tensor, path: Path) -> NoReturn:
    r""" Save an image from a tensor """
    torchvision.utils.save_image(tensor, str(path))


def _split_data():
    # if this is a substantially different dataset, give it a new name
    # this will create a whole new placeholder (Artifact) for this split
    # instead of just incrementing a version of the old data split
    run = wandb.init(project=parent_utils.get_proj_name(), job_type="data_split")

    # # create balanced train, val, test splits
    # # each count is the number of images per label
    # SPLIT_COUNTS = BALANCED_SPLITS

    # find the most recent ("latest") version of the full raw data
    # you can of course pass around programmatic aliases and not string literals
    # note: RAW_DATA_AT is defined in the previous cell—if you're running
    # just this step, you may need to hardcode it
    data_at = run.use_artifact(str(_build_proj_dir(RAW_PREFIX)) + ":latest")
    # download it locally (for illustration purposes/across hardware; you can
    # also sync/version artifacts by reference)
    data_dir = data_at.download()

    data_split_at = wandb.Artifact(str(_build_proj_dir(SPLIT_PREFIX)), type="balanced_data")
    # create a table with columns we want to track/compare
    preview_dt = wandb.Table(columns=["id", "image", "label", "split"])

    data_dir = os.path.join(data_dir, os.listdir(data_dir)[0])  # Dataset directory
    for split in os.listdir(data_dir):
        labels_path = os.path.join(data_dir, split)
        # Iterate through each class' images
        for label in os.listdir(labels_path):
            lbl_id = int(label.split("_")[0])
            img_path = os.path.join(labels_path, label)
            # Iterate through the image files
            for img_file in os.listdir(img_path):
                full_path = os.path.join(img_path, img_file)

                img_id = img_file.split("_")[-1]
                # add file to artifact by full path
                # note: pass the label to the name parameter to retain it in
                # the data structure
                data_split_at.add_file(full_path, name=os.path.join(split, label, img_id))
                # add a preview of the image
                preview_dt.add_data(img_id, wandb.Image(full_path), lbl_id, split)

    # log artifact to W&B
    data_split_at.add(preview_dt, "data_split")
    run.log_artifact(data_split_at)


def _build_proj_dir(prefix: str) -> Path:
    r""" Directory to store the project """
    flds = [prefix, config.OPTIM.lower(), f"{config.TARG_CLS}{config.POIS_CLS}"]
    return Path("_".join(flds))
