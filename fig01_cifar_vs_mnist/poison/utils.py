__all__ = [
    "ClassifierBlock",
    "NUM_WORKERS",
    "TORCH_DEVICE",
    "configure_dataset_args",
    "construct_filename",
    "get_num_usable_cpus",
    "get_proj_name",
    "load_module",
    "log_seeds",
    "save_module",
    "set_debug_mode",
    "set_random_seeds"
]

import copy
import dataclasses
import logging
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import List, NoReturn, Optional, Tuple

import numpy as np

import torch
from torch import LongTensor, Tensor
import torch.nn as nn
# noinspection PyUnresolvedReferences
from torch.optim import Adagrad, Adam, AdamW, RMSprop, SGD
from torch.utils.data import DataLoader

from .import _config as config
from . import dirs
from . import influence_utils
from .datasets import cifar
from .datasets import mnist
from .datasets.types import LearnerModule
from .losses import RiskEstimatorBase, TORCH_DEVICE
from .types import TensorGroup


LOG_LEVEL = logging.DEBUG
LOGGER_NAME = "losses"

NP_SEED = None


def get_num_usable_cpus() -> int:
    r"""
    Returns the number of usable CPUs which is less than or equal to the actual number of CPUs.
    Read the document for os.cpu_count() for more info.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


# Intelligently select number of workers
gettrace = getattr(sys, 'gettrace', None)
NUM_WORKERS = 0


def save_module(module: nn.Module, filepath: Path) -> NoReturn:
    r""" Save the specified \p model to disk """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), str(filepath))


def load_module(module: nn.Module, filepath: Path):
    r"""
    Loads the specified model in file \p filepath into \p module and then returns \p module.

    :param module: \p Module where the module on disk will be loaded
    :param filepath: File where the \p Module is stored
    :return: Loaded model
    """
    # Map location allows for mapping model trained on any device to be loaded
    module.load_state_dict(torch.load(str(filepath), map_location=TORCH_DEVICE))
    module.eval()
    return module


def construct_filename(prefix: str, out_dir: Path, file_ext: str, ex_id: Optional[int] = None,
                       add_timestamp: bool = False, add_ds_to_path: bool = True) -> Path:
    r""" Standardize naming scheme for the filename """
    fields = [config.DATASET.name.lower().replace("_", "-"),
              config.OTHER_DS.name.lower().replace("_", "-"),
              config.OPTIM.lower(),
              f"div={config.val_div()}",
              f"t-cls={config.ORIG_TARG_CLS}",
              f"p-cls={config.ORIG_POIS_CLS}",
              f"n-p={config.OTHER_CNT}",
              f"n-sep={config.NUM_SUBEPOCH}"]
    if config.DEBUG:
        fields.append("dbg")
    if ex_id is not None:
        fields.append(f"ex={ex_id}")
    if prefix:
        fields = [prefix] + fields

    if add_timestamp:
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        fields.append(time_str)

    if file_ext[0] != ".":
        file_ext = "." + file_ext
    fields[-1] += file_ext

    # Add the dataset name to better organize files
    if add_ds_to_path:
        out_dir /= config.DATASET.name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "_".join(fields)


def configure_dataset_args() -> Tuple[TensorGroup, LearnerModule]:
    r""" Manages generating the source data (if not already serialized to disk """
    if config.DATASET.is_cifar():
        cifar_dir = dirs.DATA_DIR / "CIFAR10"
        tg = cifar.load_data(cifar_dir)
        net = cifar.build_model(tg.tr_x)
    elif config.DATASET.is_mnist():
        tg = mnist.load_data(dirs.DATA_DIR)
        net = mnist.build_model(tg.tr_x)
    else:
        raise ValueError(f"Dataset generation not supported for {config.DATASET.name}")

    assert list(config.DATASET.value.dim) == list(tg.tr_x.shape[1:]), "Unexpected tensor shape"

    module = LearnerModule()
    module.set_model(module=net)

    return tg, module


def set_debug_mode(seed: int = None) -> NoReturn:
    logging.warning("Debug mode enabled")
    config.enable_debug_mode()

    set_random_seeds(seed=seed)


def set_random_seeds(seed: Optional[int] = None) -> NoReturn:
    r"""
    Sets random seeds to avoid non-determinism
    :See: https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed is None:
        seed = 545639696
    seed &= 0x7FFFFFFF  # 2^32 - 1  Need for numpy supporting 31 bit seeds only
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    log_seeds()
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = False


def log_seeds():
    r""" Log the seed information """
    logging.debug("Torch Random Seed: %d", torch.initial_seed())


@dataclasses.dataclass
class Batch:
    xs: Tensor
    lbls: LongTensor
    bd_ids: LongTensor
    ds_ids: LongTensor

    def check(self) -> NoReturn:
        r""" Encompasses checking the validity of the batch """
        assert self.xs.dtype == torch.float, "Features need to be floats"
        assert self.lbls.dtype == torch.long, "Labels unexpected type"
        assert self.bd_ids.shape == self.ds_ids.shape, "IDs tensor do not match shape"

        # ToDo remove ID checks later to save time
        influence_utils.check_bd_ids_contents(bd_ids=self.bd_ids)
        influence_utils.check_duplicate_ds_ids(ds_ids=self.ds_ids)
        msg = "Strange class labels"
        assert torch.all(0 <= self.lbls) and torch.all(self.lbls < config.N_CLASSES), msg

    def skip(self) -> bool:
        r""" If \p True, skip processing this batch """
        return self.xs.shape[0] == 0

    def __len__(self) -> int:
        r""" Returns the number of elements in the batch """
        return self.lbls.numel()


class ClassifierBlock(nn.Module):
    def __init__(self, net: nn.Module, is_pretrained: bool, estimator: RiskEstimatorBase,
                 n_subepoch: int = 1, name_prefix: str = ""):
        super().__init__()
        if n_subepoch < 1:
            raise ValueError("Number of subepoch must be greater than or equal to 1")

        self.module = copy.deepcopy(net)
        self._name_prefix = name_prefix
        self.loss = estimator

        self._is_pretrained = is_pretrained

        self.optim = None
        self.sched = None
        self._time_str = None

        # TracIn related parameters
        self._n_subepoch = n_subepoch
        self._lr_vals = None
        self._ep_ids = None  # IDs traversed in the epoch in order
        self._base_subep_ends = None  # Default subepoch endpoints split evenly
        self._subepoch_ends = None  # Location where actual subepoch ends are stored
        # Reset at start of batch and update through batch. Temporary variable of current subepoch
        self._cur_subepoch = None
        # Number of elements in batch so far
        self._n_ex_cur_epoch = None

        self.train_loss = self.num_batch = self.valid_loss = None
        self.best_loss, self._best_ep = np.inf, None

        self.tracin_stats = None

        self.alpha_i = None

        self.tracin_fin_inf = None

    def forward(self, xs: Tensor) -> Tensor:
        return self.module.forward(xs)

    def name(self) -> str:
        r""" Classifier block name derived from the loss' name """
        return self._name_prefix

    def is_poisoned(self) -> bool:
        r""" Returns \p True if poisoned data is used """
        return self.loss.is_poisoned()

    def epoch_start(self):
        r""" Configures the module for the start of an epoch """
        self.train_loss, self.num_batch = torch.zeros((), device=TORCH_DEVICE), 0

        self.valid_loss = np.inf

        # Variables storing epoch ID and subepoch divisor information
        self._ep_ids.append([])
        self._subepoch_ends.append([])
        self._lr_vals.append([])
        # Current subepoch so far
        self._cur_subepoch = 0
        # Number of training examples in current batch so far
        self._n_ex_cur_epoch = 0

        self.train()

    def _append_lr(self) -> NoReturn:
        r""" Append the learning rate value to the learner """
        self._lr_vals[-1].append(self._get_current_lr())

    def _err_check_and_process_ep_ids_data(self, ep: int) -> NoReturn:
        r"""
        Standardizes checking the epoch ID tensors. If not already done by a previous invocation to
        this function, the function also formats the IDs for easier, standardized extraction.
        """
        assert ep >= 0, "Epoch number must be non-negative"
        assert self._ep_ids is not None, "Epoch IDs list not set"
        assert ep < len(self._ep_ids), "Specified number of epochs exceeds epoch count"
        ep_ids = self._ep_ids[ep]
        # Lazily convert to a single tensor as needed
        if isinstance(ep_ids, list):
            for idx, ids_arrs in enumerate(ep_ids):
                assert len(ids_arrs) == 2, "Should be two sets of IDs per element"
                assert ids_arrs[0].shape == ids_arrs[1].shape, "Mismatch in shape of tensors"
                # Need two dimensions to merge two column vectors so cleanup of the dimensions
                # if need be
                for j, single_ids in enumerate(ids_arrs):
                    # Since all IDs checked for same shape above, ok to include this in loop
                    if len(single_ids.shape) == 2:
                        break
                    ids_arrs[j] = single_ids.unsqueeze(dim=1)
                # Concatenate columns so use dim=1
                ep_ids[idx] = torch.cat(ids_arrs, dim=1)

            # Merge into a single array and error check
            self._ep_ids[ep] = torch.cat(ep_ids, dim=0)
            assert self._ep_ids[ep].shape[1] == 2, "Should be two columns"

    def get_epoch_adv_ids(self, ep: int) -> Tensor:
        r""" Get the epoch adversarial IDs """
        self._err_check_and_process_ep_ids_data(ep=ep)
        return self._ep_ids[ep][:, 0]

    def get_epoch_dataset_ids(self, ep: int) -> Tensor:
        r""" Get the epoch dataset IDs """
        self._err_check_and_process_ep_ids_data(ep=ep)
        return self._ep_ids[ep][:, 1]

    def get_prev_lr_val(self, ep: int, subepoch: Optional[int]) -> float:
        r"""
        Gets the learning rate number from previous training

        :param ep: Epoch number
        :param subepoch: Optional subepoch value.  If not specified, then its the learning rate o
                         of the last (inc. only) subepoch
        :return: (Sub)epoch learning rate
        """
        assert ep >= 0, "Epoch number must be non-negative"
        assert ep < len(self._lr_vals), "Specified epoch number has no specified learning rate info"
        assert subepoch is None or subepoch >= 0, "Subepoch must be non-negative"

        if subepoch is None:
            subepoch = -1

        lr_vals = self._lr_vals[ep]
        assert subepoch < len(lr_vals), "Subepoch has no specified learning rate info"
        return lr_vals[subepoch]

    def _update_subepoch(self, bd_ids: Tensor, ds_ids: Tensor) -> NoReturn:
        r""" Updates the subepoch information """
        assert bd_ids.shape == ds_ids.shape, "Mismatch in subepoch shapes"
        # Stores the ids visited during the epoch in order
        self._ep_ids[-1].append([bd_ids.cpu(), ds_ids.cpu()])
        self._n_ex_cur_epoch += bd_ids.shape[0]
        # Last subepoch always runs to the end of the epoch
        if self._cur_subepoch + 1 == self._n_subepoch:
            return

        # Number of training examples consisting of the end of the subepoch if divided equally
        subepoch_end = self._base_subep_ends[self._cur_subepoch]

        # Check if current subepoch has ended
        if self._n_ex_cur_epoch >= subepoch_end:
            # Append the LR to the subepoch batch information
            self._append_lr()
            # Mark end of previous subepoch
            self._subepoch_ends[-1].append(self._n_ex_cur_epoch)
            # Save epoch after storing the epoch info
            self._save_epoch_module(ep=self._get_cur_epoch(), subepoch=self._cur_subepoch)

            self._cur_subepoch += 1

    def get_subepoch_ends(self, ep: int) -> List[int]:
        r""" List returns cumulative number of elements in each subepoch for the specified epoch """
        assert ep >= 0, "Epoch number must be positive"
        # Return a more specific error message if ep==0 to help users debug their code
        assert ep != 0, "Epoch zero by definition has no subepochs"
        assert ep < len(self._subepoch_ends), "Epoch number exceeds stored epoch ends data"

        return self._subepoch_ends[ep]

    def _get_cur_epoch(self) -> int:
        r""" Standardizes accessing the current epoch """
        return len(self._ep_ids) - 1

    def clear_serialized_models(self):
        for ep in range(0, config.NUM_EPOCH + 1):
            for subep in range(self._n_subepoch - 1):
                file_path = self._build_serialize_name(ep=ep, subepoch=subep)
                file_path.unlink()  # actually performs the deletion

            file_path = self._build_serialize_name(ep=ep, subepoch=None)
            file_path.unlink()  # actually performs the deletion

    def process_batch(self, batch_tensors: Tuple[Tensor, ...]) -> NoReturn:
        r""" Process a batch including tracking the loss and pushing the gradients """
        self.optim.zero_grad()

        batch = self.organize_batch(batch_tensors, process_mask=True)
        if batch.skip():
            return

        loss = self.loss.calc_train_loss(self.forward(batch.xs), batch.lbls)
        loss.backward()
        self.optim.step()

        self.train_loss += loss.detach()
        self.num_batch += 1

        # Must be called before scheduler step to ensure relevant LR is stored
        self._update_subepoch(bd_ids=batch.bd_ids, ds_ids=batch.ds_ids)
        if self.sched is not None:
            self.sched.step()

    # def process_mask(self, xs: Tensor, ds: Tensor, lbls: Tensor,
    #                  bd_ids: Tensor) -> Tuple[Tensor, Tensor]:
    #     r"""
    #     Extracts the ID mask and makes the requisite changes (if any) to the feature and labels
    #     vectors
    #
    #     :return: Tuple of the processed feature and label vectors respectively.  May be identical
    #              to the original, passed vectors.
    #     """
    #     # Mask returns which tensors need to be modified
    #     mask = self.loss.build_mask(ids=bd_ids)
    #
    #     # Checks if changes required so return the original tensors
    #     if self.loss.has_any(mask):
    #         # Clone prevents collision of changes across multiple parallel learners
    #         xs, lbls = xs.clone(), lbls.clone()
    #         xs[mask] += ds[mask]
    #         lbls[mask] = config.POIS_CLS
    #
    #     return xs, lbls

    def has_lr_sched(self) -> bool:
        r""" Returns \p True if the classifier block uses a scheduler """
        return self.sched is not None

    def create_optim(self, params, lr: float, wd: float):
        optim_name = config.OPTIM.upper()
        if optim_name == "SGD":
            self.optim = SGD(params, lr=lr, momentum=config.SGD_MOMENTUM, weight_decay=wd)
        elif optim_name == "ADAM":
            self.optim = Adam(params, lr=lr, weight_decay=wd)
        elif optim_name == "ADAMW":
            self.optim = AdamW(params, lr=lr, weight_decay=wd)
        elif optim_name == "RMSPROP":
            self.optim = RMSprop(params, lr=lr, weight_decay=wd)
        elif optim_name == "ADAGRAD":
            self.optim = Adagrad(params, lr=lr, weight_decay=wd)
        else:
            raise ValueError("Unknown optimizer to parse")

    def create_lr_sched(self, lr: float, train_dl: DataLoader):
        r""" Creates the Classifier learning scheduler """
        # # noinspection PyUnresolvedReferences
        # self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optim, lr, epochs=config.NUM_EPOCH,
        #                                                  steps_per_epoch=len(train_dl))
        # if self.module.is_feat_frozen():
        # if self.is_pretrained():
        #     self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim,
        #                                                             T_max=config.NUM_EPOCH)
        # else:
        use_cycle_momentum = not isinstance(self.optim, Adagrad)
        # noinspection PyUnresolvedReferences
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optim, lr,
                                                         epochs=config.NUM_EPOCH,
                                                         steps_per_epoch=len(train_dl),
                                                         cycle_momentum=use_cycle_momentum)

    def init_fit_vars(self, dl: DataLoader) -> NoReturn:
        r""" Configure the variables needed at the start of fit call """
        self._lr_vals = []
        self._ep_ids = []

        tr_ele_per_epoch = sum(bt[0].shape[0] for bt in dl)

        # Define batch size boundaries
        bs = dl.batch_size
        linspace, step = np.linspace(0, tr_ele_per_epoch, num=self._n_subepoch, retstep=True,
                                     endpoint=False, dtype=np.int64)
        if bs >= step - 10:
            logging.warning(f"Batch size {bs} but subepoch step {step}. May not use all subepoch")
        # assert bs < step - 10, f"Batch size {bs} but subepoch step {step}. May not use all subep"

        # First linspace value is 0 which is irrelevant for our purposes so skip.
        self._base_subep_ends = linspace[1:].astype("long").tolist()
        self._subepoch_ends = []

    def is_pretrained(self) -> bool:
        r""" Returns \p True if the base module is pretrained """
        return self._is_pretrained

    def calc_valid_loss(self, epoch: int, valid: DataLoader):
        r""" Calculates and stores the validation loss """
        all_scores, all_lbls = [], []

        self.eval()
        for batch_tensors in valid:
            batch = self.organize_batch(batch_tensors, process_mask=True)
            if batch.skip():
                continue

            all_lbls.append(batch.lbls)
            with torch.no_grad():
                all_scores.append(self.forward(batch.xs))

        dec_scores, labels = torch.cat(all_scores, dim=0), torch.cat(all_lbls, dim=0)
        val_loss = self.loss.calc_validation_loss(dec_scores, labels)
        self.valid_loss = abs(float(val_loss.item()))

        if self.valid_loss < self.best_loss:
            self.best_loss = self.valid_loss
            self.best_epoch = epoch

        self._save_epoch_module(ep=epoch, subepoch=None)
        self._append_lr()

    def _save_epoch_module(self, ep: int, subepoch: Optional[int]):
        r""" Serializes the (sub)epoch parameters """
        # Update the best loss if appropriate
        save_module(self, self._build_serialize_name(ep=ep, subepoch=subepoch))

    def restore_epoch_params(self, ep: int, subepoch: Optional[int]) -> NoReturn:
        r""" Restore the parameters checkpointed at the end of the specified epoch \p ep """
        assert ep >= 0, "Invalid epoch number"
        # flds = [f"Restoring {self.name()}'s epoch {ep}"]
        # if subepoch is not None:
        #     flds.append(f"subepoch {subepoch}")
        # flds.append("parameters")
        # msg = " ".join(flds)

        # logging.debug(f"Starting: {msg}")
        load_module(self, self._build_serialize_name(ep=ep, subepoch=subepoch))
        # logging.debug(f"COMPLETED: {msg}")
        self.eval()

    def restore_best(self) -> NoReturn:
        r""" Restores the best model (i.e., with the minimum validation error) """
        # msg = f"Restoring {self.name()}'s best trained model"
        # logging.debug(f"Starting: {msg}")
        load_module(self, self._build_serialize_name(ep=self.best_epoch, subepoch=None))
        # logging.debug(f"COMPLETED: {msg}")
        self.eval()

    def logger_field_info(self) -> Tuple[List[str], List[str], List[int]]:
        r"""
        Defines the block specific field names and sizes
        :return: Tuple of lists of the field names and widths (in number of characters) respectively
        """
        learner_names = [f"{self.name()}", ""]
        loss_names = [f"L-Train", f"L-Val", f"B"]

        l_name_len = len(learner_names[0]) // 2
        sizes = [max(l_name_len, 11), max(l_name_len, 11), 1]

        return learner_names, loss_names, sizes

    def epoch_log_fields(self, epoch: int):
        r""" Log the epoch information """
        return [self.train_loss / self.num_batch, self.valid_loss, self.best_epoch == epoch]

    @property
    def start_time(self) -> Optional[str]:
        r""" Define the time when training began on the module """
        return self._time_str

    @start_time.setter
    def start_time(self, value: str):
        r""" Define the time when training began on the module """
        assert self._time_str is None, "Start time already set"
        self._time_str = value

    @property
    def best_epoch(self) -> int:
        r""" Accessor for the best epoch """
        assert self._best_ep is not None, "Accessing unset best epoch"
        return self._best_ep

    @property
    def num_adv(self) -> int:
        r""" Accessor for the number of adversarial training instances used by the block """
        return self.loss.n_bd

    @best_epoch.setter
    def best_epoch(self, value: int) -> NoReturn:
        r""" Accessor for the best epoch """
        self._best_ep = value

    def _get_current_lr(self):
        r"""Gets the current learning rate"""
        assert self.optim is not None, "Getting learning rate but no optimizer"
        for param_group in self.optim.param_groups:
            return param_group['lr']

    def _build_serialize_name(self, ep: int, subepoch: Optional[int]) -> Path:
        r""" Constructs the serialized model's name """
        serialize_dir = dirs.MODELS_DIR
        serialize_dir.mkdir(parents=True, exist_ok=True)
        prefix = [f"{self.name().lower()}", f"ep={ep:03d}"]
        if subepoch is not None:
            prefix.append(f"sub={subepoch:03d}")
        return construct_filename("_".join(prefix), serialize_dir, "pth", add_timestamp=False)

    def get_module(self):
        r""" Accessor for the underlying module """
        assert self.module is not None, "Module not defined"
        if isinstance(self.module, LearnerModule):
            return self.module.get_module()
        return self.module

    def organize_batch(self, batch_tensors: Tuple[Tensor, ...], process_mask: bool,
                       verify_contents: bool = True) -> Batch:
        r"""

        :param batch_tensors: Tuple of tensors returned by the dataloader
        :param process_mask: If \p True, apply the mask to process the tensors
        :param verify_contents: If \p True, verify the batch's contents
        :return:
        """
        assert len(batch_tensors) == 4, "Unexpected batch length"
        xs, lbls, bd_ids, ds_ids = batch_tensors
        xs = xs.to(torch.device("cuda:0"))
        lbls = lbls.to(torch.device("cuda:0"))

        # noinspection PyTypeChecker
        batch = Batch(xs=xs, lbls=lbls, bd_ids=bd_ids, ds_ids=ds_ids)
        if not batch.skip() and verify_contents:
            batch.check()  # ToDo remove batch check later to save execution time
        return batch


def get_proj_name() -> str:
    r""" Returns the \p wandb project name """
    flds = [config.DATASET.name, config.OTHER_DS.name, f"{config.OTHER_CNT:03d}",
            f"{config.TARG_CLS}{config.POIS_CLS}"]
    return "_".join(flds).lower()
