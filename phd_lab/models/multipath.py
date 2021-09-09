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


class AlainFC(nn.Module):

    def __init__(self, num_classes: int, *args, **kwargs):
        super(AlainFC, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(inplace=True)
        )
        self.layers1 = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Linear(128, 128),
                    nn.ReLU(inplace=True)
                )
                for i in range(128)
             )
        )
        self.layers2 = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Linear(128, 128),
                    nn.ReLU(inplace=True)
                )
                for i in range(8)
            )
        )
        self.output = nn.Linear(128, num_classes)
        self.name = "AlainFC"

    def forward(self, x):
        out = x[:, 0, :, :].squeeze()
        out = torch.flatten(out, start_dim=1)
        first = self.first_layer(out)
        out = self.layers1(first)
        out = self.layers2(first + out)
        return self.output(out)


def alainfc(num_classes, *args, **kwargs):
    return AlainFC(num_classes)


class Pathway(nn.Module):

    def __init__(self, num_layers: int, in_planes, planes, kernel: int = 3, stride: int = 1, dilation: int = 1):
        super(Pathway, self).__init__()
        effective_kernel_size = kernel + (kernel - 1) * (dilation - 1)
        padding = effective_kernel_size // 2

        self.layers = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Conv2d(in_planes if i == 0 else planes, planes,
                              kernel_size=kernel, stride=stride if i == num_layers-1 else 1,
                              padding=padding, bias=False, dilation=dilation),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True)
                )
                for i in range(num_layers)
             )
        )

    def forward(self, x):
        return self.layers(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, num_layers_per_path: List[int], kernel_sizes_per_path: List[int], in_planes, planes, stride=1, concat: bool = False, dilations: List[int] = [1, 1]):
        super(BasicBlock, self).__init__()
        for i, (num_layers, kernel_size, dilation) in enumerate(zip(num_layers_per_path, kernel_sizes_per_path, dilations)):
            setattr(self, f"pathway{i}", Pathway(num_layers, in_planes, planes, kernel_size, stride, dilation=dilation))
        self.num_pathways = len(num_layers_per_path)
        self.concat = concat
        if self.concat:
            self.cat_conv = nn.Conv2d(planes*len(num_layers_per_path), out_channels=planes, kernel_size=(1, 1), bias=False)
            self.cat_conv_act = nn.ReLU(inplace=True)
        assert len(num_layers_per_path) == len(kernel_sizes_per_path)

    def forward(self, x):
        outputs = [getattr(self, f"pathway{i}")(x) for i in range(self.num_pathways)]
        if self.concat:
            return self.cat_conv_act(self.cat_conv(torch.cat(outputs, 1)))
        else:
            accumulator = None
            for out in outputs:
                accumulator = out if accumulator is None else accumulator + out
            return accumulator


class MPNet(nn.Module):

    def _get_max_path_idx(self, block_layout: List[int], layout_kernels: List[int]) -> int:
        expansion_per_path = [seq_length * ((kernel-1) // 2) for seq_length, kernel in zip(block_layout, layout_kernels)]
        return np.argmax(expansion_per_path)

    def __init__(self, stage_seq: List[int], block_layout: List[int], layout_kernels: List[int], concat: bool = False, num_classes: int = 10, max_path: bool = False, dilation: List[int] = [1, 1]):
        if max_path:
            max_path_idx = self._get_max_path_idx(block_layout, layout_kernels)
            layout_kernels = [layout_kernels[max_path_idx]]
            block_layout = [block_layout[max_path_idx]]
        super(MPNet, self).__init__()
        self.dilation = dilation
        self.concat = concat
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.stage1 = self._make_stage(stage_seq[0], block_layout, layout_kernels, 2, 64, 64)
        self.stage2 = self._make_stage(stage_seq[1], block_layout, layout_kernels, 2, 64, 128)
        self.stage3 = self._make_stage(stage_seq[2], block_layout, layout_kernels, 2, 128, 256)
        self.stage4 = self._make_stage(stage_seq[3], block_layout, layout_kernels, 2, 256, 512)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        final_filter_size = [i for i, e in enumerate(stage_seq) if e != 0][-1]

        self.linear = nn.Linear((2 ** (6 + final_filter_size)), num_classes)

    def _make_stage(self, num_blocks: int, num_layer_per_path: List[int], kernel_size_per_path: List[int], stride: int = 1, in_planes: int = 64, planes: int = 64):
        layers = []
        for i in range(num_blocks):
            block = BasicBlock(
                num_layer_per_path,
                kernel_size_per_path,
                in_planes if i == 0 else planes,
                planes,
                1 if i != num_blocks-1 else stride, self.concat,
                dilations=self.dilation
            )
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def mpnet36_8_1_3_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[8, 1],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet36_8_1_3_7"
    return model


def mpnet72_2_2_3_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[4,4,4,4],
        block_layout=[3,3],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet72_2_2_3_3"
    return model


def mpnet36_4_1_3_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[4, 1],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet36_4_1_3_7"
    return model


def mpnet36_4_1_3_7d3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[4, 1],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip,
        dilation=[1, 3]
    )
    model.name = "MPNet36_4_1_3_7d3"
    return model


def mpnet36_1_1_3_11d2(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[1, 1],
        layout_kernels=[3, 11],
        concat=False,
        num_classes=num_classes,
        max_path=noskip,
        dilation=[1, 2]
    )
    model.name = "MPNet36_4_1_3_11d3"
    return model


def mpnet36_1_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[1],
        layout_kernels=[3],
        concat=False,
        num_classes=num_classes,
        max_path=noskip,
        dilation=[1]
    )
    model.name = "MPNet36_1_3"
    return model


def mpnet36_1_11d2(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[1],
        layout_kernels=[11],
        concat=False,
        num_classes=num_classes,
        max_path=noskip,
        dilation=[2]
    )
    model.name = "MPNet36_1_11d3"
    return model



def mpnet36_1_7d3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[1],
        layout_kernels=[7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip,
        dilation=[3]
    )
    model.name = "MPNet36_1_7d3"
    return model


def mpnet36_4_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[4],
        layout_kernels=[3],
        concat=False,
        num_classes=num_classes,
        max_path=noskip,
        dilation=[1]
    )
    model.name = "MPNet36_4_3"
    return model


def mpnet36_1_4_3_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[1, 4],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet36_1_4_3_7"
    return model


def mpnet36_4_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[4],
        layout_kernels=[7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet36_4_7"
    return model


def mpnet36_1_2_3_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[4, 4, 4, 4],
        block_layout=[1, 2],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet36_1_2_3_7"
    return model


def mpnet29_1_2_3_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[4, 4, 3, 0],
        block_layout=[1, 2],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet29_1_2_3_7"
    return model


def mpnet36_2_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[4, 4, 4, 4],
        block_layout=[2],
        layout_kernels=[7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet36_2_7"
    return model


def mpnet36_1_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[4, 4, 4, 4],
        block_layout=[1],
        layout_kernels=[3],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet36_1_3"
    return model


def mpnet18_4_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[4],
        layout_kernels=[3],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_4_3"
    return model


def mpnet18_1_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[1],
        layout_kernels=[7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_1_7"
    return model



def mpnet36_4_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[4],
        layout_kernels=[3],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet36_4_3"
    return model


def mpnet36_8_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[8],
        layout_kernels=[3],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet36_8_3"
    return model


def mpnet36_1_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[1],
        layout_kernels=[7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet36_1_7"
    return model


def mpnet18_1_2_3_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[1, 2],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_1_2_3_7"
    return model


def mpnet18_1_4_3_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[1, 4],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_1_4_3_7"
    return model


def mpnet18_4_1_3_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[4, 1],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_4_1_3_7"
    return model


def mpnet18_8_1_3_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[8, 1],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_8_1_3_7"
    return model


def mpnet18_8_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[8],
        layout_kernels=[3],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_8_3"
    return model


def mpnet18_1_4_1_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[1, 4],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_1_4_1_7"
    return model


def mpnet18_2_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[2],
        layout_kernels=[7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_2_7"
    return model


def mpnet18_1_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[1],
        layout_kernels=[7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_1_7"
    return model



def mpnet18_1_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[1],
        layout_kernels=[3],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_1_3"
    return model


def mpnet18_1_1(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[1],
        layout_kernels=[1],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_1_1"
    return model


def mpnet18_2_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[2],
        layout_kernels=[3],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_2_3"
    return model


def mpnet18_1_2_1_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[1, 2],
        layout_kernels=[1, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_1_2_1_7"
    return model


def mpnet18_2_1(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[2],
        layout_kernels=[1],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_2_1"
    return model



def mpnet18_1_2_1_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[1, 2],
        layout_kernels=[1, 3],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_1_2_1_3"
    return model


def mpnet18_2_2_3_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[2, 2],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_2_2_3_7"
    return model


def mpnet12_2_2_3_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 1, 0],
        block_layout=[2, 2],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet12_2_2_3_7"
    return model


def mpnet12_2_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 1, 0],
        block_layout=[2],
        layout_kernels=[3],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet12_2_3"
    return model


def mpnet12_2_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 1, 0],
        block_layout=[2],
        layout_kernels=[7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet12_2_7"
    return model


def mpnet18_4_4_3_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[4, 4],
        layout_kernels=[3, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_4_4_3_7"
    return model


def mpnet18_4_4_1_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[4, 4],
        layout_kernels=[1, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_4_4_1_7"
    return model


def mpnet18_4_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[4],
        layout_kernels=[7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_4_7"
    return model


def mpnet18_4_1(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[4],
        layout_kernels=[1],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_4_1"
    return model


def mpnet18_4_3(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[1, 1, 1, 1],
        block_layout=[4],
        layout_kernels=[3],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_4_3"
    return model


def mpnet18_2_2_1_7(num_classes, noskip=False, **kwargs):
    model = MPNet(
        stage_seq=[2, 2, 2, 2],
        block_layout=[2, 2],
        layout_kernels=[1, 7],
        concat=False,
        num_classes=num_classes,
        max_path=noskip
    )
    model.name = "MPNet18_2_2_1_7"
    return model


if __name__ == '__main__':
    model = mpnet18_1_2_1_7(10, noskip=True)
    x = torch.zeros((8, 3, 32, 32))
    print(model)
    print(model(x).shape)
