from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from poison import config, logger
import poison.dirs
import poison.end_to_end
from poison.generate_results import calculate_results
import poison.learner
import poison.utils

import tr_set_analysis


def parse_args() -> Namespace:
    r""" Parse, checks, and refactors the input arguments"""
    args = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # noinspection PyTypeChecker
    args.add_argument("config_file", help="Path to the configuration file", type=Path)

    # args.add_argument("--targ", help="Overrides the target index", default=None, type=int)
    args = args.parse_args()

    if not args.config_file.exists() or not args.config_file.is_file():
        raise ValueError(f"Unknown configuration file \"{args.config_file}\"")

    # Need to update directories first as other commands rely on these directories
    logger.setup()

    config.parse(args.config_file)
    config.enable_debug_mode()
    poison.utils.set_debug_mode()
    # config.set_quiet()
    # Setup W&B last to ensure settings are correctly loaded
    poison.wandb.setup()

    # Generates the data for learning
    args.tg, args.module = poison.utils.configure_dataset_args()
    config.print_configuration()
    return args


def _main(args: Namespace):
    learners = poison.learner.train_primary_only(base_module=args.module, tg=args.tg)
    calculate_results(args.tg, erm_learners=learners)

    learners = poison.learner.train_both(base_module=args.module, tg=args.tg)
    calculate_results(args.tg, erm_learners=learners)

    # Select the target
    block = learners.get_only_block()
    poison.end_to_end.check_success(block=block, x_targ=args.tg.targ_x,
                                    y_targ=args.tg.targ_y)

    poison.end_to_end.run(tg=args.tg, block=block,
                          targ_x=args.tg.targ_x, targ_y=args.tg.targ_y)

    # Baselines
    tr_set_analysis.baselines(learners=learners, tg=args.tg,
                              targ_x=args.tg.targ_x, targ_y=args.tg.targ_y)


if __name__ == '__main__':
    _main(parse_args())
    poison.wandb.cleanup()
