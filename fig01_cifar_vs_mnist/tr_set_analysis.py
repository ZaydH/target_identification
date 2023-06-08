__all__ = [
    "baselines",
]

import logging
from typing import NoReturn, Optional

from torch import LongTensor, Tensor

from poison import config
import poison.end_to_end
import poison.influence_func
from poison.influence_utils import InfluenceMethod
import poison.learner
import poison.rep_point
import poison.gas_and_tracin
import poison.tracin_utils
from poison.types import TensorGroup
import poison.utils


def _log_example_prediction(block: poison.utils.ClassifierBlock, x: Tensor,
                            y_true: Optional[Tensor]) -> NoReturn:
    r""" Helper method to print the predicted class """
    assert x.shape[0], "Only a single training example can be logged"
    if y_true is not None:
        assert y_true.shape[0] == y_true.numel() == 1, "Y shape does not match expectation"

    x_tfms = config.get_test_tfms()(x.to(poison.utils.TORCH_DEVICE))
    y_hat = block.module.predict(x_tfms)

    msg = f"{block.name()} -- Example Predicted Label: {y_hat.item()}"
    if y_true is not None:
        msg += f", True Label: {y_true.item()}"

    logging.info(msg)


def baselines(learners: poison.learner.CombinedLearner, tg: TensorGroup,
              targ_x: Tensor, targ_y: LongTensor) -> NoReturn:
    train_dl, _ = poison.learner.create_fit_dataloader(tg=tg, use_both_classes=True)

    flds = (InfluenceMethod.REP_POINT,
            InfluenceMethod.REP_POINT_RENORM,
            )
    for method in flds:
        rep_targ_vals = poison.rep_point.calc_representer_vals(erm_learners=learners,
                                                               train_dl=train_dl,
                                                               test_x=targ_x,
                                                               method=method)
        idx = 0  # Only applies if multiple test examples
        poison.rep_point.log_representer_scores(idx, targ_y.item(), rep_targ_vals,
                                                file_prefix="adv-cls", ex_id=config.TARG_IDX,
                                                method=method, train_dl=train_dl)

    poison.utils.set_random_seeds()
    for block_name, block in learners.blocks():  # type: str, poison.utils.ClassifierBlock
        # Disable augmentation in transforms initially
        tmp_tr_dl = poison.tracin_utils.configure_train_dataloader(train_dl)
        poison.influence_func.calc(block=block, tr_dl=tmp_tr_dl, te_x=targ_x, te_y=targ_y,
                                   ex_id=config.TARG_IDX)
