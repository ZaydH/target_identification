__all__ = [
    "CombinedLearner",
    "create_fit_dataloader",
    "filter_ids",
    "train_both",
]

from abc import ABC, abstractmethod
import datetime
import dill as pk
import itertools
import logging
import time
from typing import Iterable, NoReturn, Optional, Tuple

import torch
from torch import BoolTensor, Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from . import _config as config
from . import dirs
from .logger import TrainingLogger
from .losses import Loss, ce_loss
from .types import CustomTensorDataset, LearnerParams, TensorGroup
from . import utils
from .utils import ClassifierBlock, TORCH_DEVICE


class _BaseLearner(nn.Module, ABC):
    def __init__(self, name: str):
        super(_BaseLearner, self).__init__()
        self._logger = self._train_start = None
        self._name = name

    def _configure_fit_vars(self, modules: nn.ModuleDict, train_dl: DataLoader) -> NoReturn:
        r""" Set initial values/construct all variables used in a fit method """
        # Fields that apply regardless of loss method
        learner_names, loss_names, sizes = [], [], []
        for mod_name, module in modules.items():  # type: str, ClassifierBlock
            _name, _loss_names, _size = module.logger_field_info()
            learner_names.extend(_name)
            loss_names.extend(_loss_names)
            sizes.extend(_size)

            module.init_fit_vars(dl=train_dl)

            # Initializes variables needed by the module while fitting
            lr = config.get_learner_val(mod_name, LearnerParams.Attribute.LEARNING_RATE)
            wd = config.get_learner_val(mod_name, LearnerParams.Attribute.WEIGHT_DECAY)

            params = self._get_optim_params(module)

            module.create_optim(params, lr=lr, wd=wd)
            module.create_lr_sched(lr=lr, train_dl=train_dl)

        # Always log the time in number of seconds
        learner_names.append("")
        loss_names.append("Time")
        sizes.append(10)
        self._logger = TrainingLogger(learner_names, loss_names, sizes,
                                      logger_name=utils.LOGGER_NAME)

    def _log_epoch(self, ep: int, modules: nn.ModuleDict) -> NoReturn:
        r"""
        Log the results of the epoch
        :param ep: Epoch number
        :param modules: Modules to log
        """
        flds = []
        for _, module in modules.items():
            flds.extend(module.epoch_log_fields(epoch=ep))

        flds.append(time.time() - self._train_start)
        self._logger.log(ep, flds)

    def train_start_time(self) -> str:
        r""" Returns the training start time as a string """
        assert self._train_start is not None, "Training never started"
        return datetime.datetime.fromtimestamp(self._train_start).strftime("%Y-%m-%d-%H-%M-%S")

    @abstractmethod
    def fit(self, tg: TensorGroup) -> NoReturn:
        r""" Fit all models """

    def _fit(self, modules: nn.ModuleDict, train_dl: DataLoader,
             valid_dl: DataLoader) -> NoReturn:
        r""" Fits \p modules' learners to the training and validation \p DataLoader objects """
        self._configure_fit_vars(modules, train_dl=train_dl)

        # Special handle epoch 0 since no change to model in that stage so no need to do any
        # transforms in the training dataloader
        ep = 0
        for _, module in modules.items():  # type: str, ClassifierBlock
            module.epoch_start()
            module.calc_valid_loss(epoch=ep, valid=valid_dl)
        self._log_epoch(ep, modules)

        # Handle epochs with actual updates
        for ep in range(1, config.NUM_EPOCH + 1):
            # Reset all variables tracking epoch starts
            for module in modules.values():
                module.epoch_start()

            for batch, module in itertools.product(train_dl, modules.values()):
                module.process_batch(batch)

            for module in modules.values():
                module.calc_valid_loss(epoch=ep, valid=valid_dl)

            self._log_epoch(ep, modules)

        self._restore_best_model(modules)
        self.eval()

    @staticmethod
    def _get_optim_params(module: ClassifierBlock):
        r"""
        Special function to disable weight decay of the bias (and other terms)
        See:
        https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially\
        -with-batchnorm/16994/5
        and
        https://discuss.pytorch.org/t/changing-the-weight-decay-on-bias-using-named\
        -parameters/19132/3
        """
        decay, no_decay = [], []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            # if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            # if len(param.shape) == 1 or name.endswith(".bias") or "bn" in name:
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        l2_val = config.get_learner_val(module.name(), LearnerParams.Attribute.WEIGHT_DECAY)
        return [{'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': l2_val}]

    @staticmethod
    def _restore_best_model(modules: nn.ModuleDict):
        r""" Restores the best trained model from disk """
        for _, module in modules.items():
            module.restore_best()


# noinspection PyPep8Naming
class CombinedLearner(_BaseLearner):
    r"""
    Single learner that trains multiple modules using the different quantities of poisoning
    and PN (test and train) risk estimators.
    """
    TRAIN_LOSS = ce_loss
    VALID_LOSS = ce_loss

    def __init__(self, blocks: nn.ModuleDict):
        super().__init__("CIFAR-vs-MNIST")
        self._blocks = blocks
        self.to(device=TORCH_DEVICE)

    @classmethod
    def build_std_learner(cls, base_module: nn.Module, name_prefix: str,
                          n_subep_override: Optional[int] = None) -> 'CombinedLearner':
        r""" Constructs two learners, one using poison and the other not """
        blocks = nn.ModuleDict()
        # Number of adversarial samples to test
        n_adv = config.OTHER_CNT
        n_subep_cnt = n_subep_override if n_subep_override is not None else config.NUM_SUBEPOCH
        # Create losses with different quantities of poisoning
        n_subep_lst = [n_subep_cnt] if n_adv > 0 else [1]
        for n_subep in n_subep_lst:
            loss = Loss(train_loss=cls.TRAIN_LOSS, valid_loss=cls.VALID_LOSS, n_bd=n_adv)
            block = ClassifierBlock(net=base_module, estimator=loss, name_prefix=name_prefix,
                                    is_pretrained=True, n_subepoch=n_subep)
            block.to(utils.TORCH_DEVICE)
            blocks[block.name()] = block

        return cls(blocks=blocks)

    @classmethod
    def build_repeat_learner(cls, base_module: utils.LearnerModule,
                             desc: str, repeat_id: int) -> 'CombinedLearner':
        r""" Constructs a learner that uses the filtered dataset """
        blocks = nn.ModuleDict()
        # Number of adversarial samples to test
        n_adv = config.OTHER_CNT
        n_subep = 1
        # Create losses with different quantities of poisoning
        loss = Loss(train_loss=cls.TRAIN_LOSS, valid_loss=cls.VALID_LOSS, n_bd=n_adv)
        prefix = f"repeat-{desc}-itr{repeat_id:03d}"
        block = ClassifierBlock(net=base_module, estimator=loss, name_prefix=prefix,
                                is_pretrained=True, n_subepoch=n_subep)
        block.to(utils.TORCH_DEVICE)
        blocks[block.name()] = block

        return cls(blocks=blocks)

    def fit(self, tg: TensorGroup, use_both_classes: bool = True) -> DataLoader:
        r"""
        Fit all models
        :param tg: Training & testing tensors
        :param use_both_classes: If \p True, train using both the primary and secondary classes
        :return: Training \p DataLoader
        """
        self._train_start = time.time()
        # Set the start time for each of the blocks
        for _, block in self.blocks():
            block.start_time = self.train_start_time()

        train, valid = create_fit_dataloader(tg=tg, use_both_classes=use_both_classes)
        self._fit(self._blocks, train_dl=train, valid_dl=valid)

        self.eval()
        return train

    def forward(self, x: Tensor) -> dict:
        # noinspection PyUnresolvedReferences
        return {key: block.forward(x) for key, block in self.blocks()}

    def blocks(self) -> Iterable:
        r""" Iterates through all the blocks """
        return self._blocks.items()

    def get_only_block(self) -> ClassifierBlock:
        r""" Accessor for the only block in the learner """
        assert len(self._blocks) == 1, "Learner has more than one module so unsure what to return"
        for _, block in self._blocks.items():
            return block  # noqa

    def clear_models(self):
        r""" Restores the best trained model from disk """
        for _, module in self._blocks.items():
            module.clear_serialized_models()


def create_fit_dataloader(tg: TensorGroup, use_both_classes: bool) \
        -> Tuple[DataLoader, DataLoader]:
    r"""
    Simple method that splits the positive and unlabeled sets into stratified training and
    validation \p DataLoader objects

    :param tg: TensorGroup of vectors
    :param use_both_classes: If \p True, train dataloader contains both the primary and secondary
                             class instances
    :return: Training and validation \p DataLoader objects respectively
    """
    x_tensors, y_tensors, ds_ids_tensors = [tg.tr_x], [tg.tr_y], [tg.tr_ids]
    bd_ids = torch.full_like(tg.tr_y, config.N_TRAIN)

    if use_both_classes:
        x_tensors.append(tg.bd_x)
        y_tensors.append(tg.bd_y)
        ds_ids_tensors.append(tg.bd_ids)
        # Backdoor ID numbers
        bd_bd_ids = torch.arange(tg.bd_y.numel())
        # Only mark as backdoor those examples in BD tensor from poison class
        bd_ids = torch.cat([bd_ids, bd_bd_ids], dim=0)

    x, y = torch.cat(x_tensors, dim=0), torch.cat(y_tensors, dim=0)
    ds_ids = torch.cat(ds_ids_tensors, dim=0)

    # Construct the train dataloader
    tr_ds = CustomTensorDataset([x, y, bd_ids, ds_ids], transform=config.get_train_tfms())
    tr = DataLoader(tr_ds, shuffle=True, drop_last=True, batch_size=config.BATCH_SIZE,
                    num_workers=utils.NUM_WORKERS)

    # construct the validation dataloader
    bd_ids = torch.full_like(tg.val_y, config.N_TRAIN)
    val_ds = CustomTensorDataset([tg.val_x, tg.val_y, bd_ids, tg.val_ids],
                                 transform=config.get_test_tfms())
    val = DataLoader(val_ds, shuffle=False, drop_last=False, batch_size=config.BATCH_SIZE,
                     num_workers=utils.NUM_WORKERS)

    return tr, val


def filter_ids(filt_ids: Tensor, all_ids: Tensor) -> BoolTensor:
    r""" Filter the all of the IDs to exclude the filtered IDs are included """
    # Create a mask determining which IDs to keep
    max_all_ids = torch.max(all_ids)
    keep_mask = torch.ones(max_all_ids + 1, dtype=torch.bool)
    # Remove the filtered IDs
    if filt_ids.numel() > 0:
        assert max_all_ids.item() >= torch.max(filt_ids).item(), "Some filter IDs not found"
        keep_mask[filt_ids] = False
    else:
        logging.warning("filt_ids has no elements.  Nothing filtered")

    keep_mask = keep_mask[all_ids]
    # Reduce the set of IDs
    subset_ids = all_ids[keep_mask]
    assert subset_ids.shape[0] + filt_ids.shape[0] == all_ids.shape[0], "Mismatch in IDs"
    logging.info(f"# Training IDs Filtered: {filt_ids.shape[0]}")

    return keep_mask


def train_both(base_module: nn.Module, tg: TensorGroup) -> CombinedLearner:
    r"""
    Trains a model with both CIFAR10 and MNIST data

    :param base_module: Usually untrained model used
    :param tg: Tensor groups
    :return: Collection of trained classifiers
    """
    name_prefix = "CIFAR10_and_MNIST_Joint_Model"
    return _train_base(base_module=base_module, tg=tg, use_both_classes=True,
                       name_prefix=name_prefix, overwrite_saved=True)


def train_primary_only(base_module: nn.Module, tg: TensorGroup) -> CombinedLearner:
    r"""
    Trains the learner using on just CIFAR data
    :param base_module: Usually untrained model used
    :param tg: Tensor groups
    :return: Collection of trained classifiers
    """
    name_prefix = "CIFAR10_Only_Model"
    return _train_base(base_module=base_module, tg=tg, use_both_classes=False,
                       name_prefix=name_prefix, overwrite_saved=True,
                       n_subep_override=1)


def _train_base(base_module: nn.Module, tg: TensorGroup, use_both_classes: bool,
                name_prefix: str, overwrite_saved: bool = False,
                n_subep_override: Optional[int] = None) -> CombinedLearner:
    r""" Standardizes training with and without any filtered IDs """
    full_prefix = f"{name_prefix}-fin"
    train_net_path = utils.construct_filename(full_prefix, out_dir=dirs.MODELS_DIR, file_ext="pk")

    if not train_net_path.exists() or overwrite_saved:
        learner = CombinedLearner.build_std_learner(base_module=base_module,
                                                    name_prefix=name_prefix,
                                                    n_subep_override=n_subep_override)
        learner.fit(tg=tg, use_both_classes=use_both_classes)
        learner.cpu()
        logging.info(f"Saving final {name_prefix} model...")
        with open(str(train_net_path), "wb+") as f_out:
            pk.dump(learner, f_out)

    # Load the saved module
    logging.info(f"Loading final {name_prefix} model...")
    with open(str(train_net_path), "rb") as f_in:
        learner = pk.load(f_in)  # CombinedLearner
    learner.to(utils.TORCH_DEVICE)
    return learner
