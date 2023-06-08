__all__ = [
    "cleanup",
    "setup",
]

# import logging

import wandb

from . import _config as config
from . import utils


def setup():
    r""" Setup Weights & Biases """
    if config.USE_WANDB:
        # logging.getLogger('wandb').setLevel(logging.WARNING)
        wandb.init(project=utils.get_proj_name())
        # config.print_configuration()
        utils.log_seeds()


def cleanup():
    r""" Setup Weights & Biases """
    if config.USE_WANDB:
        wandb.finish()
