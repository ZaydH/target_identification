__all__ = [
    "BATCH_SIZE",
    "DAMP",
    "DATASET",
    "HVP_BATCH_SIZE",
    "LEARNING_RATE",
    "NUM_EPOCH",
    "NUM_FF_LAYERS",
    "NUM_SUBEPOCH",
    "N_CLASSES", "N_FULL_TR", "N_TRAIN", "N_TEST",
    "ORIG_POIS_CLS",
    "ORIG_TARG_CLS",
    "OTHER_CNT",
    "OTHER_DS",
    "POIS_CLS",
    "QUIET",
    "R_DEPTH",
    "SCALE",
    "SGD_MOMENTUM",
    "TARG_CLS",
    "TARG_IDX",
    "T_REPEATS",
    "WEIGHT_DECAY",
    "enable_debug_mode",
    "get_test_tfms",
    "get_train_tfms",
    "parse",
    "print_configuration",
    "set_tfms"
]

import logging
from pathlib import Path
from typing import Callable, NoReturn, Optional, Union

from ruamel.yaml import YAML

from torch import Tensor

from .datasets.types import PoisonDataset
from .types import LearnerParams, PathOrStr, TensorGroup

DATASET = None  # type: Optional[PoisonDataset]
DATASET_KEY = "dataset"
OTHER_DS = None  # type: Optional[PoisonDataset]
OTHER_DS_KEY = "other_ds"

DEBUG = False

N_FULL_TR = -1  # Full training set size
N_TRAIN = -1  # Size of actual (filtered) training set used
N_TEST = -1

N_CLASSES = -1

NUM_FF_LAYERS = None

OPTIM = ""
SGD_MOMENTUM = 0

BATCH_SIZE = -1
NUM_EPOCH = -1
NUM_SUBEPOCH = -1
LEARNING_RATE = 1E-3
WEIGHT_DECAY = 1E-4

# Fraction of training samples used for
VALIDATION_SPLIT_RATIO = 1 / 6

TARG_IDX = None
ORIG_POIS_CLS = None
ORIG_TARG_CLS = None

POIS_CLS = None
TARG_CLS = None

QUIET = False
USE_WANDB = True

TRAIN_TFMS = None
TEST_TFMS = None

HVP_BATCH_SIZE = -1
DAMP = None
SCALE = None
R_DEPTH = None
T_REPEATS = None

OTHER_CNT = 0

LEARNER_CONFIGS = dict()


def parse(config_file: PathOrStr) -> NoReturn:
    r""" Parses the configuration """
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Unable to find config file {config_file}")
    if not config_file.is_file():
        raise FileExistsError(f"Configuration {config_file} does not appear to be a file")

    with open(str(config_file), 'r') as f_in:
        all_yaml_docs = YAML().load_all(f_in)

        base_config = next(all_yaml_docs)
        _parse_general_settings(base_config)
        # _parse_learner_specific_settings(all_yaml_docs)


def _parse_general_settings(config) -> NoReturn:
    r"""
    Parses general settings for the learning configuration including the dataset, priors, positive
    & negative class information.  It also extracts general learner information
    """
    module_dict = _get_module_dict()
    for key, val in config.items():
        if key.lower() in (DATASET_KEY, OTHER_DS_KEY):
            ds_name = val.upper()
            try:
                module_dict[key.upper()] = PoisonDataset[ds_name]
            except KeyError:
                raise ValueError(f"Unknown dataset {ds_name}")
        # elif key.lower() == ATTACK_KEY:
        #     _parse_attack_mode(file_val=val)
        else:
            key = key.upper()
            if key not in module_dict:
                raise ValueError(f"Unknown configuration field \"{key}\"")
            module_dict[key] = val

    # if DATASET.is_newsgroups():
    #     _parse_newsgroups(module_dict)
    global ORIG_POIS_CLS, ORIG_TARG_CLS
    assert ORIG_POIS_CLS is None and ORIG_TARG_CLS is None, "Original values not set by user"
    ORIG_POIS_CLS = POIS_CLS
    ORIG_TARG_CLS = TARG_CLS
    # Sanity checks nothing is out of whack
    _verify_configuration()


def _get_module_dict() -> dict:
    r""" Standardizes construction of the module dictionary """
    return globals()


def _verify_configuration() -> NoReturn:
    r""" Sanity checks the configuration """
    if DATASET is None:
        raise ValueError("A dataset must be specified")

    if VALIDATION_SPLIT_RATIO <= 0 or VALIDATION_SPLIT_RATIO >= 1:
        raise ValueError("Validation split ratio must be in range (0,1)")

    supported_optims = ["adam", "sgd", "adamw", "adagrad", "rmsprop"]
    if OPTIM is None or not OPTIM:
        raise ValueError("Configuration does not specify an optimizer")
    if OPTIM.lower() not in supported_optims:
        raise ValueError(f"{OPTIM} is an unknown or unsupported optimizer")

    if LEARNING_RATE <= 0:
        raise ValueError("Learning rate must be positive")
    if NUM_EPOCH <= 0:
        raise ValueError("Number of training epochs must be positive")
    if NUM_SUBEPOCH <= 0:
        raise ValueError("Number of training epochs must be positive")
    if WEIGHT_DECAY < 0:
        raise ValueError("Weight decay must be non-negative")

    if OTHER_CNT < 0:
        raise ValueError("Other count must be positive")

    # noinspection PyTypeChecker
    if NUM_FF_LAYERS is None or NUM_FF_LAYERS < 0:
        raise ValueError("Number of FF layers must be non-negative")


def print_configuration(log: Callable = logging.info) -> NoReturn:
    r""" Print the configuration settings """
    log(f"Dataset: {DATASET.name}")
    log(f"Other Dataset: {OTHER_DS.name}")

    log(f"Number Training Examples: {N_TRAIN:,}")
    log(f"# Other Class Examples: {OTHER_CNT:,}")
    log(f"Inductive Test Set Size: {N_TEST:,}")

    log(f"# FF Layers: {NUM_FF_LAYERS}")

    log(f"Optimizer: {OPTIM.lower()}")
    log(f"Batch Size: {BATCH_SIZE}")
    log(f"# Epoch: {NUM_EPOCH}")
    log(f"# Subepoch: {NUM_SUBEPOCH}")
    log(f"Learning Rate: {LEARNING_RATE:.0E}")
    log(f"Weight Decay: {WEIGHT_DECAY:.0E}")

    log(f"Target Class True Label: {ORIG_TARG_CLS}")
    log(f"Target Class Binary Label: {TARG_CLS}")
    log(f"Target Example Index: {TARG_IDX}")
    log(f"Poison Class True Label: {ORIG_POIS_CLS}")
    log(f"Poison Class Binary Label: {POIS_CLS}")

    log(f"Quiet Mode: {QUIET}")

    log(f"HVP Batch Size: {HVP_BATCH_SIZE}")
    log(f"HVP Recursive Depth: {R_DEPTH}")
    log(f"HVP T Repeat: {T_REPEATS}")
    log(f"HVP Damp Param: {DAMP}")
    log(f"HVP Scale Param: {SCALE:.0E}")


def reset_learner_settings() -> NoReturn:
    r""" DEBUG ONLY.  Reset the settings specific to individual learners/loss functions """
    global LEARNER_CONFIGS
    LEARNER_CONFIGS = dict()


def set_layer_counts(ff_layers: Optional[int] = None,
                     sigma_layers: Optional[int] = None) -> NoReturn:
    r""" Set the number of learner layers """
    assert ff_layers is not None or sigma_layers is not None, "Must set at least one layer count"

    if ff_layers is not None:
        global NUM_FF_LAYERS
        NUM_FF_LAYERS = ff_layers


def set_ds_sizes(n_full_tr: Optional[int] = None, n_train: Optional[int] = None,
                 # n_max_adv: Optional[int] = None,
                 n_test: Optional[int] = None) -> NoReturn:
    r""" Optionally sets the dataset sizes """
    global N_FULL_TR, N_TRAIN, N_TEST
    if n_full_tr is not None:
        N_FULL_TR = n_full_tr
    if n_train is not None:
        N_TRAIN = n_train
    # if n_max_adv is not None:
    #     N_ADV = n_max_adv
    if n_test is not None:
        N_TEST = n_test


def set_all_ds_sizes(n_full_tr: int, tg: TensorGroup) -> NoReturn:
    r""" Sets the all dataset sizes using a TensorGroup """
    n_train = tg.tr_x.shape[0] + tg.val_x.shape[0] + tg.te_bd_x.shape[0]
    n_test = tg.te_cl_x.shape[0] + tg.te_bd_x.shape[0]
    set_ds_sizes(n_full_tr=n_full_tr, n_train=n_train, n_test=n_test)


def override_num_subepoch(n_subep: int) -> NoReturn:
    r""" Overrides the number of number of subepoch in the configuration """
    assert n_subep > 0, "Number of poison override cannot be negative"
    global NUM_SUBEPOCH
    NUM_SUBEPOCH = n_subep
    logging.warning(f"Overriding the number of subepoch to {NUM_SUBEPOCH}")


def override_targ_idx(targ_idx: Union[Tensor, int]) -> NoReturn:
    r""" Overrides the number of poison in the configuration """
    if isinstance(targ_idx, Tensor):
        assert targ_idx.numel() == 1, "Only single targets are allowed"
        targ_idx = targ_idx.item()
    assert targ_idx >= 0, "Target index must be positive"
    global TARG_IDX
    TARG_IDX = targ_idx
    logging.warning(f"Overriding the target index to {TARG_IDX}")


def set_num_classes(n_classes: int) -> NoReturn:
    r""" Sets the number of training classes """
    global N_CLASSES
    N_CLASSES = n_classes


def set_quiet() -> NoReturn:
    r""" Enables quiet mode """
    global QUIET
    QUIET = True


def set_rand_cls_labels(targ_lbl: int, adv_lbl: int) -> NoReturn:
    r""" Sets the random class index """
    assert targ_lbl != adv_lbl, "True and adversarial label cannot match"

    global POIS_CLS, TARG_CLS
    TARG_CLS = targ_lbl
    POIS_CLS = adv_lbl


def set_rand_cls_idx(rand_idx: int) -> NoReturn:
    r""" Sets the random class index """
    assert rand_idx >= 0, "Test set index cannot be negative"

    global TARG_IDX
    TARG_IDX = rand_idx
    logging.info(f"Random Test Index: {TARG_IDX}")


def enable_debug_mode() -> NoReturn:
    r""" Enables debug mode for the learner """
    global DEBUG
    DEBUG = True


def has_tfms() -> bool:
    r""" Returns \p True if the module has a normalize transform """
    return TRAIN_TFMS is not None


def get_train_tfms():
    r""" Accessor for the training set transforms """
    assert has_tfms(), "Getting non-existent train transforms"
    return TRAIN_TFMS


def get_test_tfms():
    r""" Accessor for the training set transforms """
    assert has_tfms() and TEST_TFMS is not None, "Getting non-existent test transforms"
    return TEST_TFMS


def set_tfms(train_tfms, test_tfms) -> NoReturn:
    r""" Sets the training and test transforms """
    assert train_tfms is not None and test_tfms is not None, "No transforms specified"
    module_dict = _get_module_dict()
    for ds in ("train", "test"):
        tfms = locals()[f"{ds}_tfms"]
        module_dict[f"{ds.upper()}_TFMS"] = tfms


def val_div() -> int:
    r""" Get the validation divider """
    return int(round(1 / VALIDATION_SPLIT_RATIO))


# noinspection PyUnusedLocal
def get_learner_val(learner_name: str, param: LearnerParams.Attribute) -> Union[int, float]:
    r""" Gets learner specific values """
    if param == LearnerParams.Attribute.WEIGHT_DECAY:
        return WEIGHT_DECAY
    if param == LearnerParams.Attribute.LEARNING_RATE:
        return LEARNING_RATE
    raise ValueError("Parameter not supported")


def update_labels(targ_cls: int, pois_cls: int) -> NoReturn:
    r""" Update the target and poison labels """
    global TARG_CLS, POIS_CLS
    TARG_CLS, POIS_CLS = targ_cls, pois_cls
