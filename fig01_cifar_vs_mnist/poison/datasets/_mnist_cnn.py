__all__ = [
    "Model",
]
r""""
Adapted from the Robustness Against Backdoors (RAB) repository.

See: https://github.com/AI-secure/Robustness-Against-Backdoor-Attacks
"""

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .types import PoisonLearner


class Model(PoisonLearner):
    def __init__(self):
        super(Model, self).__init__(n_classes=1)

        # Note: noqa below due to type resolve errors when ints used instead of tuples for params
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)  # noqa
        # self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)  # noqa
        # self.bn2 = nn.BatchNorm2d(num_features=32)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(32*4*4, 512)
        # self.output = nn.Linear(512, 1)

    # def unfix_pert(self,):
    #     del self.fixed_pert

    # def fix_pert(self, sigma, hash_num):
    #     assert not hasattr(self, 'fixed_pert')
    #     rand = np.random.randint(2**32-1)
    #     np.random.seed(hash_num)
    #     self.fixed_pert = torch.FloatTensor(1,1,28,28).normal_(0, sigma)
    #     if self.gpu:
    #         self.fixed_pert = self.fixed_pert.cuda()
    #     np.random.seed(rand)

    def conv_only(self) -> nn.Sequential:
        return nn.Sequential(self.conv1,
                             self.max_pool,
                             # self.bn1,
                             self.conv2,
                             # self.bn2,
                             )

    def forward(self, x: Tensor, penu: bool = False, block: bool = False) -> Tensor:
        assert not block, "Block mode not currently supported"
        # B = x.size()[0]

        # if hasattr(self, 'fixed_pert'):
        #     x = x + self.fixed_pert

        out = x
        out = self.max_pool(F.relu(self.conv1(out)))
        # out = self.bn1(out)
        out = self.max_pool(F.relu(self.conv2(out)))
        # out = self.bn2(out)
        out = self.flatten(out)
        out = self.fc_first(out)

        if penu:
            return out
        out = self.linear(out)
        return out

    # def loss(self, pred, label):
    #     if self.gpu:
    #         label = label.cuda()
    #     label = label.float()
    #     return F.binary_cross_entropy_with_logits(pred, label)
