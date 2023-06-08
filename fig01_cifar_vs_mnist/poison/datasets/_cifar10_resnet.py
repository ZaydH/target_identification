__all__ = [
    "ResNet9"
]

import torch
from torch import Tensor
import torch.nn as nn

from .types import PoisonLearner
from .. import _config as config


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # noqa
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.res1 = nn.Sequential(conv_block(dim, dim), conv_block(dim, dim))

    def forward(self, xb):
        return self.res1(xb) + xb


class ResNet9(PoisonLearner):
    def __init__(self, in_channels, num_classes, tr_p_dropout: float = 0, te_p_dropout: float = 0):
        assert 0 <= tr_p_dropout < 1, "Invalid training dropout rate. Must be in range [0,1)"
        assert 0 <= te_p_dropout < 1, "Invalid test dropout rate. Must be in range [0,1)"

        super().__init__(n_classes=num_classes, tr_p_dropout=tr_p_dropout,
                         te_p_dropout=te_p_dropout)

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = ResBlock(128)

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = ResBlock(512)

        self.max_pool = nn.MaxPool2d(4 if not config.DATASET.is_mnist() else 3)
        self.flatten = nn.Flatten()

        # Only use to get decoder only parameters
        self.decoder = nn.Sequential(self.conv1, self.conv2, self.res1, self.conv3,
                                     self.conv4, self.res2, self.max_pool)

    def conv_only(self) -> nn.Sequential:
        return self.decoder

    def forward(self, x: Tensor, penu: bool = False, block: bool = False):
        feat_list = []
        out = self.conv1(x)
        out = self.apply_dropout(out)

        out = self.conv2(out)
        out = self.apply_dropout(out)

        out = self.res1(out)
        out = self.apply_dropout(out)

        out = self.conv3(out)
        # if block:
        #     feat_list.append(out)
        out = self.apply_dropout(out)

        out = self.conv4(out)
        if block:
            feat_list.append(out)
        out = self.apply_dropout(out)

        out = self.res2(out)
        out = self.max_pool(out)
        out = self.flatten(out)

        if len(self.fc_first) > 0:
            feat_list.append(out)
            out = self.fc_first(out)
        if block:
            feat_list.append(out)
            return feat_list
        if penu:
            return out
        out = self.apply_dropout(out)

        out = self.linear(out)
        return out
