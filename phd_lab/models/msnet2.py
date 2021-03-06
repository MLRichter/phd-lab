from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional


class LinearTransitionLayer(nn.Module):

    def __init__(self, num_input: int, scale_facotrs: List[float], in_features: int, out_features: int):
        super().__init__()
        if num_input != len(scale_facotrs):
            raise ValueError(f"Every Input needs its own scale factor, "
                             f"found {len(scale_facotrs)} scale factors and "
                             f"{num_input} inputs")
        self.scale_factors = scale_facotrs
        self.num_input = num_input
        self.compress = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        )

    def forward(self, *x):
        resized = []
        for xi, scale_factor in zip(x, self.scale_factors):
            rszd = functional.interpolate(xi, scale_factor=scale_factor)
            resized.append(rszd)
        concat = torch.cat(resized, 1)
        out = self.compress(concat)
        return out


def conv(in_planes: int, planes: int, k_size: int, downsample, act) -> nn.Module:
    return nn.Sequential(*[
        nn.Conv2d(in_planes, planes, k_size, stride=1 if not downsample else 2, padding=1),
        nn.BatchNorm2d(planes),
        act()
    ])


class ConvolutionalBlock(nn.Module):

    def __init__(self, in_planes: int, planes: int, k_size: int = 3, downsample: bool = False, act = nn.ReLU):
        super().__init__()
        self.conv1 = conv(in_planes, planes, k_size, False, act)
        self.conv2 = conv(planes, planes, k_size, downsample, act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MSNet2(nn.Module):

    def __init__(self, design_resolution: int, num_classes: int = 10, low_res: bool = False,
                 blocks: List[int] = [2,2,2,2], trans: bool = True, trans1: bool = True,
                 act = nn.ReLU):
        super().__init__()
        self.design_resolution = design_resolution
        self.num_classes = num_classes
        self.low_res = low_res
        self.trans = trans
        self.trans1 = trans1
        if not self.low_res:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                act(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                act(inplace=True)
            )

        # Stage 1
        self.stage1 = nn.Sequential(
            *[ConvolutionalBlock(64, 64, 3, act=act) for i in range(blocks[0])]
        )
        # Stage 2
        self.stage2 = nn.Sequential(
            *[ConvolutionalBlock(64 if i == 0 else 128, 128, 3, i == 0, act=act) for i in range(blocks[1])]

        )
        # Stage 3
        if trans and trans1:
            self.transition1 = LinearTransitionLayer(2, [0.5, 1.0], 64+128, out_features=256)
        else:
            print("Skipping Trans1")
            self.transition1 = LinearTransitionLayer(1, [1.0], 128, out_features=256)

        self.stage3 = nn.Sequential(
            *[ConvolutionalBlock(256, 256, 3, i == 0, act=act) for i in range(blocks[2])]

        )
        # Stage 4
        if trans:
            self.transition2 = LinearTransitionLayer(3, [0.25, 0.5, 1.0], 64+128+256, 512)
        else:
            self.transition2 = LinearTransitionLayer(1, [1.0], 256, 512)
        self.stage4 = nn.Sequential(
            *[ConvolutionalBlock(512, 512, 3, i == 0, act=act) for i in range(blocks[3])]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        if self.trans and self.trans1:
            t1 = self.transition1(x1, x2)
        else:
            t1 = self.transition1(x2)
        x3 = self.stage3(t1)
        if self.trans:
            t2 = self.transition2(x1, x2, x3)
        else:
            t2 = self.transition2(x3)
        x4 = self.stage4(t2)
        pooled = self.pool(x4)
        pooled = torch.flatten(pooled, 1)
        out = self.classifier(pooled)
        return out


def msnet22_nt(pretrained: bool = False, progress: bool = True, input_size: Tuple[int, int, int] = None, **kwargs):
    model = MSNet2(224, num_classes=kwargs["num_classes"], low_res=True, blocks=[2, 2, 3, 3], trans=False)
    model.name = "MSNet22_NT"
    return model


def msnet22(pretrained: bool = False, progress: bool = True, input_size: Tuple[int, int, int] = None, **kwargs):
    model = MSNet2(224, num_classes=kwargs["num_classes"], low_res=True, blocks=[2, 2, 3, 3], trans=True)
    model.name = "MSNet22"
    return model


def msnet22_swish(pretrained: bool = False, progress: bool = True, input_size: Tuple[int, int, int] = None, **kwargs):
    model = MSNet2(224, num_classes=kwargs["num_classes"], low_res=True, blocks=[2, 2, 3, 3], trans=True, act=nn.Hardswish)
    model.name = "MSNet22_Swish"
    return model


def msnet22fpn(pretrained: bool = False, progress: bool = True, input_size: Tuple[int, int, int] = None, **kwargs):
    model = MSNet2(224, num_classes=kwargs["num_classes"], low_res=True, blocks=[2, 2, 3, 3], trans=True, trans1=False)
    model.name = "MSNet22FPN"
    return model