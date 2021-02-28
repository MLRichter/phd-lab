from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional


class TransitionLayer(nn.Module):

    def __init__(self, num_input: int, scale_facotrs: List[float], in_features: int, out_features: int):
        super().__init__()
        if num_input != len(scale_facotrs):
            raise ValueError(f"Every Input needs its own scale factor, "
                             f"found {len(scale_facotrs)} scale factors and "
                             f"{num_input} inputs")
        self.scale_factors = scale_facotrs
        self.num_input = num_input
        self.compress = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features)
        )

    def forward(self, *x):
        resized = []
        for xi, scale_factor in zip(x, self.scale_factors):
            rszd = functional.interpolate(xi, scale_factor=scale_factor)
            resized.append(rszd)
        concat = torch.cat(resized, 1)
        out = self.compress(concat)
        return out


def conv(in_planes: int, planes: int, k_size: int, downsample) -> nn.Module:
    return nn.Sequential(*[
        nn.Conv2d(in_planes, planes, k_size, stride=1 if not downsample else 2, padding=1),
        nn.BatchNorm2d(planes),
        nn.ReLU()
    ])


class ConvolutionalBlock(nn.Module):

    def __init__(self, in_planes: int, planes: int, k_size: int = 3, downsample: bool = False):
        super().__init__()
        self.conv1 = conv(in_planes, planes, k_size, False)
        self.conv2 = conv(planes, planes, k_size, downsample)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MSNet(nn.Module):

    def __init__(self, design_resolution: int, num_classes: int = 10, low_res: bool = False, blocks: List[int] = [2,2,2,2], trans: bool = True):
        super().__init__()
        self.design_resolution = design_resolution
        self.num_classes = num_classes
        self.low_res = low_res
        if not self.low_res:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

        # Stage 1
        self.stage1 = nn.Sequential(
            *[ConvolutionalBlock(64, 64, 3) for _ in range(blocks[0])]
        )
        # Stage 2
        self.stage2 = nn.Sequential(
            *[ConvolutionalBlock(64 if i == 0 else 128, 128, 3, i == 0) for i in range(blocks[1])]

        )
        # Stage 3
        if trans:
            self.transition1 = TransitionLayer(2, [0.5, 1.0], 64+128, out_features=128)
        else:
            self.transition1 = None
        self.stage3 = nn.Sequential(
            *[ConvolutionalBlock(128 if i == 0 else 256, 256, 3, i == 0) for i in range(blocks[2])]

        )
        # Stage 4
        if trans:
            self.transition2 = TransitionLayer(3, [0.25, 0.5, 1.0], 64+128+256, 256)
        else:
            self.transition2 = None
        self.stage4 = nn.Sequential(
            *[ConvolutionalBlock(256 if i == 0 else 512, 512, 3, i == 0) for i in range(blocks[3])]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        if self.transition1 is not None:
            t1 = self.transition1(x1, x2)
        else:
            t1 = x2
        x3 = self.stage3(t1)
        if self.transition2 is not None:
            t2 = self.transition2(x1, x2, x3)
        else:
            t2 = x3
        x4 = self.stage4(t2)
        pooled = self.pool(x4)
        pooled = torch.flatten(pooled, 1)
        out = self.classifier(pooled)
        return out


def msnet18nt(pretrained: bool = False, progress: bool = True, input_size: Tuple[int, int, int] = None, **kwargs):
    model = MSNet(224, num_classes=kwargs["num_classes"], low_res=False, blocks=[2, 2, 2, 2], trans=False)
    model.name = "MSNet_NT"
    return model


def msnet18(pretrained: bool = False, progress: bool = True, input_size: Tuple[int, int, int] = None, **kwargs):
    model = MSNet(224, num_classes=kwargs["num_classes"], low_res=False, blocks=[2, 2, 2, 2], trans=True)
    model.name = "MSNet18"
    return model


def msnet18_ns(pretrained: bool = False, progress: bool = True, input_size: Tuple[int, int, int] = None, **kwargs):
    model = MSNet(224, num_classes=kwargs["num_classes"], low_res=True, blocks=[2, 2, 2, 2], trans=True)
    model.name = "MSNet18_NS"
    return model


def msnet18_ntns(pretrained: bool = False, progress: bool = True, input_size: Tuple[int, int, int] = None, **kwargs):
    model = MSNet(224, num_classes=kwargs["num_classes"], low_res=True, blocks=[2, 2, 2, 2], trans=False)
    model.name = "MSNet18_NSNT"
    return model
