__all__ = [
    "build_model",
    "load_data",
]

import dill as pk
from pathlib import Path
from typing import NoReturn, Optional

import torch
from torch import BoolTensor, LongTensor, Tensor
import torch.nn as nn
import torch.nn.functional as F  # noqa
# noinspection PyProtectedMember
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from . import _cifar10_resnet as cifar10_resnet  # noqa
from . import utils
from . import wandb_utils
from .. import _config as config
from ..types import TensorGroup
from .. import utils as parent_utils

DS_SIZE = 50000
TEST_ELE_PER_CLS = 1000

N_CIFAR_CLASSES = 10

CIFAR_MIN = 0
CIFAR_MAX = 1

LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def build_model(x: Tensor) -> nn.Module:
    r""" Construct the model used for MNIST training """
    # model = cifar10_cnn.Model()
    model = cifar10_resnet.ResNet9(in_channels=3, num_classes=1)
    return utils.build_model(model=model, x=x)


def download_data(cifar_dir: Path) -> Path:
    r""" Loads the CIFAR10 dataset """
    tfms = [torchvision.transforms.ToTensor()]

    tensors_dir = cifar_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    for is_training in [True, False]:
        # Path to write the processed tensor
        file_path = tensors_dir / f"{'training' if is_training else 'test'}.pth"
        if file_path.exists():
            continue

        # noinspection PyTypeChecker
        ds = torchvision.datasets.cifar.CIFAR10(cifar_dir,
                                                transform=torchvision.transforms.Compose(tfms),
                                                train=is_training, download=True)
        dl = DataLoader(ds, batch_size=config.BATCH_SIZE, num_workers=0,
                        drop_last=False, pin_memory=False, shuffle=False)

        # Construct the full tensors
        all_x, all_y = [], []
        for xs, ys in dl:
            all_x.append(xs.cpu())
            all_y.append(ys.cpu())
        # X is already normalized in range [0,1] so no need to normalize
        all_x, all_y = torch.cat(all_x, dim=0).float(), torch.cat(all_y, dim=0).long()
        # Write the pickle file
        with open(str(file_path), "wb+") as f_out:
            torch.save((all_x, all_y), f_out)
    return tensors_dir


def _set_transforms():
    r"""
    Configures the training and test/validation transforms used.  Based on
    https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    """
    # normalize_tfm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    tfms_tr = transforms.Compose([
    ])

    tfms_val = transforms.Compose([
    ])
    config.set_tfms(train_tfms=tfms_tr, test_tfms=tfms_val)


def _construct_tensorgroup(cifar_dir: Path) -> TensorGroup:
    cifar_tensors_dir = download_data(cifar_dir)
    # Prune dataset by classes
    paths = utils.prune_datasets(base_dir=cifar_dir, data_dir=cifar_tensors_dir)
    # Transfer the vectors to the tensor groups
    tg = TensorGroup()
    utils.populate_main_class(tg=tg, paths=paths)

    # Add MNIST to the dataset
    _add_mnist_tensors(tg=tg, cifar_dir=cifar_dir)
    utils.extract_target_example(tg)

    return tg


def _add_mnist_tensors(tg: TensorGroup, cifar_dir: Path) -> NoReturn:
    r""" Add the MNIST tensors to the \p TensorGroup """
    # Download MNIST images inside CIFAR to isolate from MNIST-based experiments
    mnist_base = cifar_dir / config.OTHER_DS.name

    from . import mnist
    mnist_tensors_dir = mnist.download_mnist_dataset(dest=mnist_base,
                                                     ds_name=config.OTHER_DS.name)

    paths = utils.prune_datasets(base_dir=mnist_base, data_dir=mnist_tensors_dir)
    bd_path, te_bd_path = paths[0], paths[-1]

    _load_mnist_examples_from_file(tg=tg, ds_name="bd", path=bd_path, cnt=config.OTHER_CNT,
                                   offset_ids=True)
    _load_mnist_examples_from_file(tg=tg, ds_name="te_bd", path=te_bd_path)


def _load_mnist_examples_from_file(tg: TensorGroup, ds_name: str, path: Path,
                                   cnt: Optional[int] = None, offset_ids: bool = False) -> NoReturn:
    r""" Load MNIST examples only from the poison class """
    x, y, ids = torch.load(path)  # type: Tensor, LongTensor, LongTensor
    mask = y == config.POIS_CLS  # type: BoolTensor
    # # DEBUG ONLY -- Use MNIST examples from both classes
    # mask = (y == config.POIS_CLS).logical_or(y == config.TARG_CLS)  # type: BoolTensor
    x, y, ids = x[mask], y[mask], ids[mask]

    if cnt is not None:
        rand_idx = torch.randperm(torch.sum(mask).item(), dtype=torch.long)[:cnt]
        x, y, ids = x[rand_idx], y[rand_idx], ids[rand_idx]

    # Normalize x features to range [0, 1]
    x = _convert_mnist_to_cifar(x=x)
    # Ensure unique IDs for MNIST examples
    if offset_ids:
        ids += DS_SIZE
    utils.update_tensorgroup(tg=tg, ds_name=ds_name, x=x, y=y, ids=ids)


def _convert_mnist_to_cifar(x: Tensor) -> Tensor:
    r""" Converts an MNIST tensor to match the dimensions of CIFAR """
    x = x.unsqueeze(dim=1)  # Add color channel
    x = x.repeat(1, 3, 1, 1)  # Convert color channel to appear 3 times
    x = F.pad(input=x, pad=4 * [2], value=0)
    assert all(x_dim == c_dim for x_dim, c_dim in zip(x.shape[1:], config.DATASET.value.dim))

    from . import mnist
    x = x.float() / mnist.NORMALIZE_FACTOR
    return x


def _calc_num_tr_ex() -> int:
    r""" Calculates and the total number of training examples """
    from . import mnist
    return DS_SIZE + mnist.DS_SIZE



def _run_load(cifar_dir: Path) -> TensorGroup:
    r""" Need to run twice to ensure valid random seed """
    parent_utils.set_random_seeds()

    _set_transforms()
    tg_pkl_path = parent_utils.construct_filename("bk-tg", out_dir=cifar_dir,
                                                  file_ext="pkl", add_ds_to_path=False)

    tg = _construct_tensorgroup(cifar_dir=cifar_dir)
    with open(tg_pkl_path, "wb+") as f_out:
        pk.dump(tg, f_out)
    wandb_utils.upload_data(tg=tg, labels=LABELS)

    config.set_all_ds_sizes(n_full_tr=_calc_num_tr_ex(), tg=tg)
    utils.binarize_labels(tg=tg)
    config.override_targ_idx(targ_idx=tg.targ_ids.item())
    config.set_num_classes(n_classes=10)

    parent_utils.set_random_seeds()
    return tg


def load_data(cifar_dir: Path) -> TensorGroup:
    r""" Loads the CIFAR10 dataset """
    # Download data only
    _construct_tensorgroup(cifar_dir=cifar_dir)

    # One twice to ensure the same expected data
    tg = _run_load(cifar_dir=cifar_dir)
    return tg
