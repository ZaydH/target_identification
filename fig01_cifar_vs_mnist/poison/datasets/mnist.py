__all__ = [
    "DS_SIZE",
    "NORMALIZE_FACTOR",
    "build_model",
    "load_data",
]

import dill as pk
import logging
from pathlib import Path
import re
from typing import NoReturn, Optional

import torch
from torch import BoolTensor, LongTensor, Tensor
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from . import _mnist_cnn as mnist_cnn
from . import utils
from . import wandb_utils
from .. import _config as config
from .. import dirs
from ..types import TensorGroup
from .. import utils as parent_utils

DS_SIZE = 60000
MNIST_FF_HIDDEN_DIM = 512
NORMALIZE_FACTOR = 255
NUM_CLASSES = 10


def build_model(x: Tensor) -> nn.Module:
    r""" Construct the model used for MNIST training """
    model = mnist_cnn.Model()
    return utils.build_model(model=model, x=x, hidden_dim=MNIST_FF_HIDDEN_DIM)


def download_mnist_dataset(dest: Path, ds_name: str = "") -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    flds = []
    for train, ds_name in ((True, "training"), (False, "test")):
        ds = torchvision.datasets.MNIST(root=dest.parent, download=True, train=train)
        logging.debug(f"Downloaded dataset MNIST ({ds_name})")

        # Export the dataset information
        flds.append((f"{ds_name}.pt", ds.data, ds.targets))

    dest /= "processed"
    dest.mkdir(parents=True, exist_ok=True)
    for base_name, x, y in flds:
        full_name = dest / base_name
        if full_name.exists():
            continue
        torch.save((x, y), full_name)
    return dest


def _set_transforms():
    r"""
    Configures the training and test/validation transforms used. No transforms specified for MNIST
    """
    tfms_tr = transforms.Compose([])

    tfms_val = transforms.Compose([])
    config.set_tfms(train_tfms=tfms_tr, test_tfms=tfms_val)


def _calc_num_tr_ex() -> int:
    r""" Calculates and the total number of training examples """
    from . import cifar
    return DS_SIZE + cifar.DS_SIZE


def _construct_tensorgroup(mnist_dir: Path) -> TensorGroup:
    tensors_dir = download_mnist_dataset(mnist_dir)
    # Prune dataset by classes
    paths = utils.prune_datasets(base_dir=mnist_dir, data_dir=tensors_dir)
    # Transfer the vectors to the tensor groups
    tg = TensorGroup()
    utils.populate_main_class(tg=tg, paths=paths, normalize_func=_normalize_func)

    # Add MNIST to the dataset
    _add_cifar_tensors(tg=tg, mnist_dir=mnist_dir)

    utils.extract_target_example(tg)
    return tg


def _normalize_func(x: Tensor) -> Tensor:
    x = x.float() / NORMALIZE_FACTOR
    x = x.unsqueeze(dim=1)
    return x


def _add_cifar_tensors(tg: TensorGroup, mnist_dir: Path) -> NoReturn:
    r""" Add the MNIST tensors to the \p TensorGroup """
    # Download MNIST images inside CIFAR to isolate from MNIST-based experiments
    cifar_base = mnist_dir / config.OTHER_DS.name

    from . import cifar
    cifar_tensors_dir = cifar.download_data(cifar_dir=cifar_base)

    paths = utils.prune_datasets(base_dir=cifar_base, data_dir=cifar_tensors_dir)
    bd_path, te_bd_path = paths[0], paths[-1]

    _load_cifar_examples_from_file(tg=tg, ds_name="bd", path=bd_path, cnt=config.OTHER_CNT,
                                   offset_ids=True)
    _load_cifar_examples_from_file(tg=tg, ds_name="te_bd", path=te_bd_path)


def _load_cifar_examples_from_file(tg: TensorGroup, ds_name: str, path: Path,
                                   cnt: Optional[int] = None, offset_ids: bool = False) -> NoReturn:
    r""" Load CIFAR examples only from the poison class """
    x, y, ids = torch.load(path)  # type: Tensor, LongTensor, LongTensor
    mask = y == config.POIS_CLS  # type: BoolTensor
    # # DEBUG ONLY -- Use MNIST examples from both classes
    # mask = (y == config.POIS_CLS).logical_or(y == config.TARG_CLS)  # type: BoolTensor
    x, y, ids = x[mask], y[mask], ids[mask]

    if cnt is not None:
        rand_idx = torch.randperm(torch.sum(mask).item(), dtype=torch.long)[:cnt]
        x, y, ids = x[rand_idx], y[rand_idx], ids[rand_idx]

    x = _convert_cifar_to_mnist(x=x)
    # Ensure unique IDs for MNIST examples
    if offset_ids:
        ids += DS_SIZE
    utils.update_tensorgroup(tg=tg, ds_name=ds_name, x=x, y=y, ids=ids)


def _convert_cifar_to_mnist(x: Tensor) -> Tensor:
    r""" Converts an CIFAR tensor to match the format of MNIST """
    x = x[:, :, 2:30, 2:30]  # Convert feature dimension -- shave off outer border

    x = transforms.Grayscale(num_output_channels=1)(x)
    assert all(x_dim == c_dim for x_dim, c_dim in zip(x.shape[1:], config.DATASET.value.dim))

    return x


def load_data(base_dir: Path) -> TensorGroup:
    r""" Loads the MNIST dataset """
    _set_transforms()
    mnist_dir = base_dir / config.DATASET.value.name
    tg_pkl_path = parent_utils.construct_filename("bk-tg", out_dir=mnist_dir,
                                                  file_ext="pkl", add_ds_to_path=False)

    if not tg_pkl_path.exists():
        tg = _construct_tensorgroup(mnist_dir=mnist_dir)

        with open(tg_pkl_path, "wb+") as f_out:
            pk.dump(tg, f_out)
        wandb_utils.upload_data(tg=tg, labels=list(range(NUM_CLASSES)))

    with open(tg_pkl_path, "rb") as f_in:
        tg = pk.load(f_in)  # type: TensorGroup

    config.set_all_ds_sizes(n_full_tr=_calc_num_tr_ex(), tg=tg)
    utils.binarize_labels(tg=tg)
    config.set_num_classes(n_classes=NUM_CLASSES)
    config.override_targ_idx(targ_idx=tg.targ_ids.item())
    return tg
