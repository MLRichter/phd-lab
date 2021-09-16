'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ThreeNet(nn.Module):

    def __init__(self, num_classes: int, middle_layer: int = 128, *args, **kwargs):
        super(ThreeNet, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(3*32*32, middle_layer * 2),
            nn.ReLU(inplace=True)
        )
        self.layers1 = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Linear(middle_layer * 2, middle_layer),
                    nn.ReLU(inplace=True)
                )
             )
        )
        self.output = nn.Linear(middle_layer, num_classes)
        self.name = f"ThreeNet{middle_layer}"

    def forward(self, x):
        out = x[:, :, :, :].squeeze()
        out = torch.flatten(out, start_dim=1)
        first = self.first_layer(out)
        out = self.layers1(first)
        return self.output(out)


def ThreeNet8(num_classes, *args, **kwargs):
    return ThreeNet(num_classes, middle_layer=8)


def ThreeNet16(num_classes, *args, **kwargs):
    return ThreeNet(num_classes, middle_layer=16)


def ThreeNet32(num_classes, *args, **kwargs):
    return ThreeNet(num_classes, middle_layer=32)


def ThreeNet64(num_classes, *args, **kwargs):
    return ThreeNet(num_classes, middle_layer=64)


def ThreeNet128(num_classes, *args, **kwargs):
    return ThreeNet(num_classes, middle_layer=128)


def ThreeNet256(num_classes, *args, **kwargs):
    return ThreeNet(num_classes, middle_layer=256)


def ThreeNet512(num_classes, *args, **kwargs):
    return ThreeNet(num_classes, middle_layer=512)


def ThreeNet1024(num_classes, *args, **kwargs):
    return ThreeNet(num_classes, middle_layer=1024)