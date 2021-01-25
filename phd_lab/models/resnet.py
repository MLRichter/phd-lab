from typing import List, Union

import torch
import torch.nn.functional as F
import torchvision
from math import floor
from operator import mul
from ..experiments.utils.pca_layers import Conv2DPCALayer, LinearPCALayer
from torchvision.models import ResNet, vgg19_bn as vgg19_orig, vgg16_bn as vgg16_orig, resnet34 as resnet34_orig, \
    resnet152 as resnet152_orig

PCA = False
PRETRAINED = False

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def pretrainedResNet151(num_classes, *args, **kwargs):
    net = resnet152_orig(pretrained=True, num_classes=num_classes)
    net.name = 'ResNet152'
    return net

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(int(in_planes), int(out_planes), kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(int(in_planes), int(out_planes), kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, thresh=.99, centering=False, noskip=False, nodownsampling=False):
        super(BasicBlock, self).__init__()
        self.noskip = noskip
        self.nodownsampling = nodownsampling
        self.thresh = thresh
        self.centering = centering
        self.conv1 = conv3x3(inplanes, planes, stride)
        if PCA:
            self.convpca1 = Conv2DPCALayer(planes, threshold=thresh, centering=centering)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if PCA:
            self.convpca2 = Conv2DPCALayer(planes, threshold=thresh, centering=centering)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if PCA:
            out = self.convpca1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if PCA:
            out = self.convpca2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.noskip and not (self.nodownsampling and self.downsample is not None):
            out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, threshold=.999, centering=False, noskip=False, nodownsampling=False):
        super(Bottleneck, self).__init__()
        self.noskip = noskip
        self.conv1 = conv1x1(inplanes, planes)
        if PCA:
            self.conv1PCA = Conv2DPCALayer(planes, threshold, centering)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        if PCA:
            self.conv2PCA = Conv2DPCALayer(planes, threshold, centering)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        if PCA:
            self.conv3PCA = Conv2DPCALayer(planes, threshold, centering)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if PCA:
            out = self.conv1PCA(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if PCA:
            out = self.conv2PCA(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if PCA:
            out = self.conv3PCA(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if not self.noskip:
            out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, thresh=.999, centering=False,
                 noskip=False, scale_factor=1, nodownsampling=False, highway=False, noskip_by_layer=None, disable_early_downsampling=False, disable_early_pooling=False, **kwargs):
        super(ResNet, self).__init__()
        self.disable_early_pooling = disable_early_pooling
        if len(layers) <= 4:
            for _ in range(len(layers), 9):
                layers.append(None)

        if noskip:
            self.noskip_by_layer = [True] * len(layers)
        elif noskip_by_layer is None:
            self.noskip_by_layer = [False] * len(layers)
        else:
            self.noskip_by_layer = noskip_by_layer

        self.nodownsampling = nodownsampling
        self.highway = highway
        self.noskip = noskip
        self.inplanes = 64 // scale_factor
        self.thresh = thresh
        self.centering = centering
        self.conv1 = nn.Conv2d(3, int(64 // scale_factor), kernel_size=7, stride=2 if not disable_early_downsampling else 1, padding=3,
                               bias=False)
        if PCA:
            self.conv1pca = Conv2DPCALayer(64, threshold=thresh, centering=centering)
        self.bn1 = nn.BatchNorm2d(int(64 // scale_factor))
        self.relu = nn.ReLU(inplace=True)
        if not self.disable_early_pooling:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64 // scale_factor), layers[0], threshold=thresh, centering=centering,
                                       noskip=self.noskip_by_layer[0])
        self.layer2 = None if layers[1] is None else self._make_layer(block, int(128 // scale_factor), layers[1],
                                                                      stride=2, threshold=thresh, centering=centering,
                                                                      nodownsampling=nodownsampling,
                                                                      noskip=self.noskip_by_layer[1])
        self.layer3 = None if layers[2] is None else self._make_layer(block, int(256 // scale_factor), layers[2],
                                                                      stride=2, threshold=thresh, centering=centering,
                                                                      nodownsampling=nodownsampling,
                                                                      noskip=self.noskip_by_layer[2])
        self.layer4 = None if layers[3] is None else self._make_layer(block, int(512 // scale_factor), layers[3],
                                                                      stride=2, threshold=thresh, centering=centering,
                                                                      nodownsampling=nodownsampling,
                                                                      noskip=self.noskip_by_layer[3])
        self.layer5 = None if layers[4] is None else self._make_layer(block, int(512 // scale_factor), layers[4],
                                                                      stride=2, threshold=thresh, centering=centering,
                                                                      nodownsampling=nodownsampling,
                                                                      noskip=self.noskip_by_layer[4])
        self.layer6 = None if layers[5] is None else self._make_layer(block, int(512 // scale_factor), layers[5],
                                                                      stride=2, threshold=thresh, centering=centering,
                                                                      nodownsampling=nodownsampling,
                                                                      noskip=self.noskip_by_layer[5])
        self.layer7 = None if layers[6] is None else self._make_layer(block, int(512 // scale_factor), layers[6],
                                                                      stride=2, threshold=thresh, centering=centering,
                                                                      nodownsampling=nodownsampling,
                                                                      noskip=self.noskip_by_layer[6])
        self.layer8 = None if layers[7] is None else self._make_layer(block, int(512 // scale_factor), layers[7],
                                                                      stride=2, threshold=thresh, centering=centering,
                                                                      nodownsampling=nodownsampling,
                                                                      noskip=self.noskip_by_layer[7])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(self.get_fully_connected_units(layers) // scale_factor) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _get_skip_flag(self, index: int, noskip: Union[bool, List[bool]]) -> bool:
        if isinstance(noskip, bool):
            return noskip
        else:
            return noskip[index]

    def _make_layer(self, block, planes, blocks, stride=1, threshold=.999, centering=False, nodownsampling=False, noskip=False):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion) and not (noskip or self.nodownsampling):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample if not noskip or not self.nodownsampling else None, threshold, centering,
                  noskip=self._get_skip_flag(0, noskip) or self.nodownsampling, nodownsampling=nodownsampling))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if self.highway:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, 1),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = None
            layers.append(block(self.inplanes, planes, noskip=self._get_skip_flag(i, noskip), nodownsampling=nodownsampling, downsample=downsample))

        return nn.Sequential(*layers)

    def _compute_num_filters(self, layer):
        return min(64 * (2 ** layer), 512)

    def _last_not_none(self, l):
        result = -1
        for i, elem in enumerate(l):
            if elem is None:
                return result
            result = i
        return result

    def get_fully_connected_units(self, layers):
        return self._compute_num_filters(self._last_not_none(layers))

    def forward(self, x):
        x = self.conv1(x)
        if PCA:
            x = self.conv1pca(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.disable_early_pooling:
            x = self.maxpool(x)

        x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        if self.layer3 is not None:
            x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        if self.layer5 is not None:
            x = self.layer5(x)
        if self.layer6 is not None:
            x = self.layer6(x)
        if self.layer7 is not None:
            x = self.layer7(x)
        if self.layer8 is not None:
            x = self.layer8(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18highway(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], highway=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18Highway'
    return model

def resnet18nodownsample(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], nodownsampling=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18NoDownsampling'
    return model


def resnet18_ep0(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], disable_early_pooling=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_EP0'
    return model


def resnet18_ep00(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], disable_early_pooling=True, disable_early_downsampling=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_EP00'
    return model


def resnet18_ep01(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], disable_early_downsampling=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_EP01'
    return model


def resnet18_noskip_ep01(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip=True, disable_early_downsampling=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18NoSkip_EP01'
    return model


def resnet18_noskip_ep00(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip=True, disable_early_pooling=True, disable_early_downsampling=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18NoSkip_EP00'
    return model

def resnet18_noskip_ep0(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip=True, disable_early_pooling=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18NoSkip_EP0'
    return model



def resnet18_nr1_1(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[True, False, False, False], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR1_1'
    return model


def resnet18_nr1_2(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[False, True, False, False], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR1_2'
    return model


def resnet18_nr1_3(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[False, False, True, False], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR1_3'
    return model


def resnet18_nr1_4(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[False, False, False, True], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR1_4'
    return model


def resnet18_nr2_12(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[True, True, False, False], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR2_12'
    return model


def resnet18_nr2_13(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[True, False, True, False], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR2_13'
    return model


def resnet18_nr2_14(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[True, False, False, True], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR2_14'
    return model


def resnet18_nr2_23(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[False, True, True, False], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR2_23'
    return model


def resnet18_nr2_24(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[False, True, False, True], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR2_24'
    return model


def resnet18_nr2_34(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[False, False, True, True], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR2_34'
    return model


def resnet18_nr3_123(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[True, True, True, False], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR3_123'
    return model


def resnet18_nr3_124(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[True, True, False, True], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR3_124'
    return model


def resnet18_nr3_234(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip_by_layer=[False, True, True, True], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_NR3_234'
    return model


def resnet18noskip(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18NoSkip'
    return model

def resnet34_nr1(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], noskip_by_layer=[True, False, False, False], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    model.name = 'ResNet34_NR1'

    return model

def resnet34_nr2(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], noskip_by_layer=[True, True, False, False], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    model.name = 'ResNet34_NR2'

    return model

def resnet34_nr32(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], noskip_by_layer=[True, True, [True, True, False, False, False, False], False], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    model.name = 'ResNet34_NR32'

    return model


def resnet34noskip(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs["noskip"] = True
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    model.name = 'ResNet34NoSkip'

    return model


def resnet18_S(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], scale_factor=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_S'
    return model


def resnet18_XS(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], scale_factor=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_XS'
    return model


def resnet18_XXS(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], scale_factor=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_XXS'
    return model


def resnet34_S(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], scale_factor=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    model.name = 'ResNet34_S'

    return model


def resnet34_XS(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], scale_factor=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    model.name = 'ResNet34_XS'

    return model


def resnet34_XXS(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], scale_factor=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    model.name = 'ResNet34_XXS'

    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18'
    return model


def resnet18_bottleneck(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_Bottleneck'
    return model

def resnet18noskip_dspl2(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs["noskip"] = True
    model = ResNet(BasicBlock, [4, 4, None, None], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18Noskip_DSPL2'
    return model


def resnet18noskip_dspl3(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs["noskip"] = True
    model = ResNet(BasicBlock, [2, 3, 3, None], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18NoSkip_DSPL3'
    return model


def resnet18_dspl2(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [4, 4, None, None], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_DSPL2'
    return model


def resnet18_dp00_dspl2(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [4, 4, None, None], disable_early_pooling=True, disable_early_downsampling=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_dp00_DSPL2'
    return model


def resnet18noskip_dp00_dspl2(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs["noskip"] = True
    model = ResNet(BasicBlock, [4, 4, None, None], disable_early_pooling=True, disable_early_downsampling=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18NoSkip_dp00_DSPL2'
    return model


def resnet18_dspl3(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 3, 3, None], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_DSPL3'
    return model


def resnet18_dspl5(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 2, 2, 2, None, None, None], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_DSPL5'
    return model


def resnet18_dspl6(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1, 2, 2, None, None], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_DSPL6'
    return model


def resnet18_dspl8(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1, 1, 1, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_DSPL8'
    return model


def resnet18_l05_w05(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], scale_factor=2, **kwargs)
    model.name = 'ResNet18(length=05 width=05)'
    return model


def resnet18_l2_w1(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    model.name = 'ResNet18(length=2 width=1)'
    return model


def resnet36_l2_w2(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [4, 4, 4, 4], scale_factor=0.5, **kwargs)
    model.name = 'ResNet18(length=2 width=2)'
    return model


def resnet18_early(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 1, 5], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_Early'
    return model


def resnet18_late(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [5, 1, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_Late'
    return model


def resnet18_wide(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], scale_factor=0.5, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_Wide'
    return model


def resnet18_thin(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [4, 4, 4, 4], scale_factor=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18_Thin'
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    model.name = 'ResNet34'

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    model.name = 'ResNet50'

    return model


def resnet50noskip(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = ResNet(Bottleneck, [3, 4, 6, 3], noskip=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    model.name = 'ResNet50NoSkip'

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    model.name = 'ResNet101'
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    model.name = 'ResNet152'
    return model
