__all__ = [
    "binarize_labels",
    "binom_sample",
    "download_file",
    "extract_target_example",
    "filter_classes",
    "get_class_mask",
    "populate_main_class",
    "prune_datasets",
    "shuffle_tensorgroup",
    "update_tensorgroup",
]

import logging
from pathlib import Path

import numpy as np
import requests
from typing import Callable, List, NoReturn, Optional, Tuple, Union

import torch
from torch import LongTensor, Tensor
from torch import nn
import torch.distributions as distributions

from .types import NEG_LABEL, POS_LABEL
from .. import _config as config
from ..types import TensorGroup


def binom_sample(prior: float, n: int) -> int:
    r""" Binomial distribution sample """
    binom = distributions.Binomial(n, torch.tensor([prior]))
    return int(binom.sample())


def multinomial_sample(n: int, p_vec: Tensor) -> Tensor:
    r""" Multinomial distribution sample """
    assert p_vec.shape[0] > 0, "Multinomial size doesn't make sense"

    n_per_category = distributions.Multinomial(n, p_vec).sample().int()

    assert p_vec.shape == n_per_category.shape, "Dimension mismatch"
    assert int(n_per_category.sum().item()) == n, "Number of elements mismatch"
    return n_per_category


def download_file(url: str, file_path: Path) -> None:
    r""" Downloads the specified file """
    CHUNK_SIZE = 128 * 1024  # BYTES  # noqa

    if file_path.exists():
        logging.info(f"File \"{file_path}\" already downloaded. Skipping...")
        return

    # Store the download file to a temporary directory
    tmp_file = file_path.parent / f"tmp_{file_path.stem}.download"

    msg = f"Downloading file at \"{url}\""
    logging.info(f"Starting: {msg}...")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(str(tmp_file), 'wb+') as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    logging.info(f"COMPLETED: {msg}")

    msg = f"Renaming temporary file \"{tmp_file}\" to \"{file_path}\""
    logging.info(f"Starting: {msg}...")
    tmp_file.rename(file_path)
    logging.info(f"COMPLETED: {msg}")

    assert file_path.exists(), "Specified file path does not exist"


def build_model(model, x: Tensor, hidden_dim: Optional[int] = None) -> nn.Module:
    r""" Constructs the linear layer """
    x_tfm = config.get_test_tfms()(x[:1])
    model.eval()
    with torch.no_grad():
        model.build_fc(x=x_tfm, hidden_dim=hidden_dim)
    return model


def get_class_mask(y: Tensor, cls_lst: Optional[Union[List[int], int]] = None) -> Tensor:
    r""" Removes all classes except the target and poison classes """
    if cls_lst is None:
        cls_lst = [config.TARG_CLS, config.POIS_CLS]
    if isinstance(cls_lst, int):
        cls_lst = [cls_lst]

    cls_mask = torch.full(y.shape, fill_value=False, dtype=torch.bool)
    for i in cls_lst:
        cls_mask |= (y == i)
    return cls_mask


def filter_classes(x: Tensor, y: LongTensor,
                   ids: LongTensor) -> Tuple[Tensor, LongTensor, LongTensor]:
    r""" Returns objects based on the """
    cls_mask = get_class_mask(y=y)
    return x[cls_mask], y[cls_mask], ids[cls_mask]


def prune_datasets(base_dir: Path, data_dir: Path) -> List[Path]:
    r""" Reduce the training and test sets based on a fixed divider of the ordering """
    # Location to store the pruned data
    prune_dir = base_dir / "pruned"
    prune_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    # div = int(round(1 / config.VALIDATION_SPLIT_RATIO))
    for i in range(3):
        is_train, is_val = i == 0, i == 1
        # Build the path for the data to store
        if is_train:
            dest_fname = f"training_div={config.val_div()}.pth"
        elif is_val:
            dest_fname = f"val_div={config.val_div()}.pth"
        else:
            dest_fname = f"test.pth"
        # Construct the complete filename and if it exists,
        paths.append(prune_dir / dest_fname)
        if paths[-1].exists():
            continue

        # Load the complete source data
        base_fname = "training" if is_train or is_val else "test"
        # Support two different file extensions
        for file_ext in (".pth", ".pt"):
            path = data_dir / (base_fname + file_ext)
            if not path.exists():
                continue
            with open(path, "rb") as f_in:
                x, y = torch.load(f_in)
            y_np = y.numpy()
            ord_ids = torch.from_numpy(np.argsort(y_np, kind="stable")).long()
            x, y = x[ord_ids], y[ord_ids]
            break
        else:
            base_combo = data_dir / base_fname
            raise ValueError(f"Unable to find processed tensor with base name \"{base_combo}\"")

        # Add an ID vector that details the original ID number
        ids = torch.arange(x.shape[0], dtype=torch.long)  # type: Tensor # noqa
        if is_train or is_val:
            mask = (ids % config.val_div()) == 0
            if is_train:
                mask = ~mask
        else:
            mask = torch.full(y.shape, True, dtype=torch.bool)  # noqa
        x, y, ids = x[mask], y[mask], ids[mask]

        with open(str(paths[-1]), "wb+") as f_out:
            torch.save((x, y, ids), f_out)

    return paths


def shuffle_tensorgroup(tg: TensorGroup) -> NoReturn:
    r""" Shuffle the tensor group elements """
    for ds in ("tr", "val", "test"):
        n_el = tg.__getattribute__(f"{ds}_ids").numel()
        # Randomize the indices
        shuffled_idx = torch.randperm(n_el)
        assert n_el == shuffled_idx.numel()
        for suffix in ("x", "d", "y", "adv_y", "ids"):
            name = f"{ds}_{suffix}"
            tensor = tg.__getattribute__(name)
            if tensor is None:
                continue

            shuffled = tensor[shuffled_idx]
            tg.__setattr__(name, shuffled)


def binarize_labels(tg: TensorGroup) -> NoReturn:
    r""" Binarize the labels """
    # Extract all the labels
    for fld in dir(tg):
        if not fld.endswith("_y"):
            continue
        lbls = tg.__getattribute__(fld)
        mask = lbls == config.ORIG_POIS_CLS
        lbls[mask] = POS_LABEL
        lbls[~mask] = NEG_LABEL

    config.update_labels(targ_cls=NEG_LABEL, pois_cls=POS_LABEL)


def extract_target_example(tg: TensorGroup) -> None:
    r""" Extracts the target example """
    mask = tg.te_bd_y == config.POIS_CLS
    # Select only the elements that have the matching class and then select the target from it
    idx = torch.randint(torch.sum(mask).item(), (1,)).item()
    for tensor_name in ("x", "y", "ids"):
        te_name = f"te_bd_{tensor_name}"
        filt_tensor = tg.__getattribute__(te_name)
        filt_tensor = filt_tensor[mask]  # Only examples from poison class

        targ_name = f"targ_{tensor_name}"
        tg.__setattr__(targ_name, filt_tensor[idx:idx + 1])


def update_tensorgroup(tg: TensorGroup, ds_name: str, x: Tensor, y: LongTensor,
                       ids: LongTensor) -> NoReturn:
    r""" Add the specified tensors to the \p TensorGroup """
    tg.__setattr__(f"{ds_name}_x", x)
    tg.__setattr__(f"{ds_name}_y", y)
    tg.__setattr__(f"{ds_name}_ids", ids)


def populate_main_class(tg: TensorGroup, paths: List[Path],
                        normalize_func: Optional[Callable] = None) -> NoReturn:
    r""" Populate the main class into the \p TensorGroup object """
    for ds_name in ("tr", "val", "te_cl"):
        x, y, ids = torch.load(paths[0])
        paths.pop(0)  # Remove from the front of the list
        # Filter to only the classes of interest
        x, y, ids = filter_classes(x=x, y=y, ids=ids)
        x = x.float()
        if normalize_func is not None:
            x = normalize_func(x)
        update_tensorgroup(tg=tg, ds_name=ds_name, x=x, y=y, ids=ids)
