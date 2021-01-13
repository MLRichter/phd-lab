import torch
import torch.nn as nn
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


def pretrainedVGG16(num_classes, *args, **kwargs):
    net = vgg16_orig(pretrained=PRETRAINED, num_classes=num_classes)
    net.name = 'VGG16'
    return net


def pretrainedVGG19(num_classes, *args, **kwargs):
    net = vgg19_orig(pretrained=PRETRAINED, num_classes=num_classes)
    net.name = 'VGG19'
    return net

cfg = {
    'V': [64, 'M'],
    'VS': [32, 'M'],
    'W': [64, 'M', 128, 'M'],
    'WS': [32, 'M', 64, 'M'],
    'X': [64, 'M', 128, 'M', 256, 'M'],
    'XS': [32, 'M', 64, 'M', 128, 'M'],
    'Y': [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'YS': [32, 'M', 64, 'M', 128, 'M', 256, 'M'],
    'YXS': [16, 'M', 32, 'M', 64, 'M', 128, 'M'],
    'Z': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'ZXS': [16, 'M', 32, 'M', 64, 'M', 128, 'M', 128, 'M'],
    'ZXXS': [8, 'M', 16, 'M', 32, 'M', 64, 'M', 64, 'M'],
    'ZXXXS': [4, 'M', 8, 'M', 16, 'M', 32, 'M', 32, 'M'],
    'ZS': [32, 'M', 64, 'M', 128, 'M', 256, 'M', 256, 'M'],
    'AS': [32, 'M', 64, 'M', 126, 126, 'M', 256, 256, 'M', 256, 256, 'M'],
    'AXS': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'AXXS': [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'AXXXS': [4, 'M', 8, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M'],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'BS': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'BXS': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'BXXS': [8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'BXXXS': [4, 4, 'M', 8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M'],

    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B_Strides': [64, (64, 2), 128, (128, 2), 256, (256, 2), 512, (512, 2), 512, (512, 2)],
    'B_Early': [64, 64, 128, 128, 256, 256, 'M', 512, 'M', 512, 'M', 512, 'M', 512, 'M'],
    'B_Late': [64, 'M', 64, 'M', 128, 'M', 128, 'M', 256, 256, 512, 512, 512, 512, 'M'],
    'B_Wide': [128, 'M', 256, 'M', 512, 'M', 1024, 'M', 1024, 'M'],
    'B_Late_Wide': [64, 'M', 64, 'M', 128, 'M', 128, 'M', 2560, 'M'],
    'B_Thin': [32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256,
               256, 256, 'M'],

    'Blin': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'DS': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'DXXXS': [4, 4, 'M', 8, 8, 'M', 16, 16, 16, 'M', 32, 32, 32, 'M', 32, 32, 32, 'M'],
    'DXXS': [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 64, 64, 64, 'M'],
    'DXS': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_P': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 512, 'M', 512, 512, 512, 512, 512, 'M'],
    'D_P2': [64, 'M', 64, 'M', 128, 'M', 128, 'M', 256, 256, 256, 512,  512, 512, 512, 512, 512, 'M'],
    'D_AP1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512, 'M'],
    'D_AP2': [64, 64, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_AP3': [64, 64, 128, 128, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_AP4': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_AP5': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    'D_AP6': [64, 64, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_AP7': [64, 64, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

    'ResNet':  [(64, 2, 7), 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    'ResNet2':  [(64, 2, 7), 'M', 128, 128, 128, 128, (256, 2), 256, 256, 256, (512, 2), 512, 512, 512, (512, 2),  512, 512, 512],
    'E':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'EP-1':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'EP-2':  [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 512, 512, 512, 'M', 512, 512, 512,'M', 512, 512, 512],
    'EP': [64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'EP1': [64, 64, 64, 'M', 128, 128, 128, 256, 'M', 256, 256, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'EP2': [64, 64, 64, 128, 128, 'M', 128, 256, 256, 256, 512, 'M', 512, 512, 512, 512, 512, 'M'],
    'ER': [64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M'],
    'ER2': [128, 128, 128, 128, 128, 128, 'M', 256, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 'M'],
    'ER3': [256, 256, 256, 256, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512, 512, 512, 'M'],
    'E2': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'E3': [128, 128, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'E4': [256, 256, 256, 256, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'E5': [512, 256, 256, 256, 'M', 512, 256, 256, 'M', 512, 512, 1024, 'M'],
    'E6': [512, 256, 256, 256, 512, 256, 256, 512, 512, 1024],
    'E7': [512, 256, 256, 256, 512, 256, 256, 512, 512, 1024],
    'ES': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
    'EXS': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 128, 128, 128, 128, 'M'],
    'EXXS': [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M', 64, 64, 64, 64, 'M'],
    'EXXXS': [4, 4, 'M', 8, 8, 'M', 16, 16, 16, 16, 'M', 32, 32, 32, 32, 'M', 32, 32, 32, 32, 'M'],
}


def vggresnet(*args, **kwargs):
    """VGG 19-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = VGG(make_layers(cfg['ResNet']), **kwargs)
    model.name = "VGGResNet"
    return model


def vggresnet2(*args, **kwargs):
    """VGG 19-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = VGG(make_layers(cfg['ResNet']), **kwargs)
    model.name = "VGGResNet2"
    return model

def vggresnet3(*args, **kwargs):
    """VGG 19-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = VGG(make_layers(cfg['ResNet']), init_weights=False, **kwargs)
    model.name = "VGGResNet3"
    return model

def vgg19r3(*args, **kwargs):
    """VGG 19-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EP-1']), **kwargs)
    model.name = "VGG19R3"
    return model

def vgg19r4(*args, **kwargs):
    """VGG 19-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EP-2']), **kwargs)
    model.name = "VGG19R4"
    return model


def vgg19p0(*args, **kwargs):
    """VGG 19-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EP']), **kwargs)
    model.name = "VGG19PO"
    return model


def vgg19p1(*args, **kwargs):
    """VGG 19-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EP1']), **kwargs)
    model.name = "VGG19P1"
    return model


def vgg19p2(*args, **kwargs):
    """VGG 19-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EP2']), **kwargs)
    model.name = "VGG19P2"
    return model



def vgg19r0(*args, **kwargs):
    """VGG 19-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ER']), **kwargs)
    model.name = "VGG19R0"
    return model


def vgg19r1(*args, **kwargs):
    """VGG 19-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ER2']), **kwargs)
    model.name = "VGG19R1"
    return model


def vgg19r2(*args, **kwargs):
    """VGG 19-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ER3']), **kwargs)
    model.name = "VGG19R2"
    return model


def vgg19r2d3(*args, **kwargs):
    """VGG 19-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ER3'], dilation=3), **kwargs)
    model.name = "VGG19R2D3"
    return model


def vgg16AlteredPooling1(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D_AP1']), **kwargs)
    model.name = "AlteredPooling1_VGG16"
    return model


def vgg16AlteredPooling2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D_AP2']), **kwargs)
    model.name = "AlteredPooling2_VGG16"
    return model


def vgg16AlteredPooling3(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D_AP3']), **kwargs)
    model.name = "AlteredPooling3_VGG16"
    return model


def vgg16AlteredPooling4(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D_AP4']), **kwargs)
    model.name = "AlteredPooling4_VGG16"
    return model


def vgg16AlteredPooling5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D_AP5']), **kwargs)
    model.name = "AlteredPooling5_VGG16"
    return model


def vgg16AlteredPooling6(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D_AP6']), **kwargs)
    model.name = "AlteredPooling6_VGG16"
    return model


def vgg16_ap7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D_AP7']), **kwargs)
    model.name = "VGG16_AP7"
    return model


def vggO8(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E5']), final_filter=1024, linear_layer=1024, pool_size=6, **kwargs)
    model.name = "VGGo8"
    return model


def vggO6_100(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E5']), final_filter=1024, linear_layer=1024, **kwargs)
    model.name = "VGGo6_100"
    return model


def vggO6_1000(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E5']), final_filter=1024, linear_layer=1024, **kwargs)
    model.name = "VGGo6_1000"
    return model


def vggO6_100_regression(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E5']), final_filter=1024, linear_layer=1024, regress=True, **kwargs)
    model.name = "VGGo6_100"
    return model


def vggO7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E7']), final_filter=1024, linear_layer=1024)
    model.name = "VGGo7"
    return model


def vggO6(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E6']), final_filter=1024)
    model.name = "VGGo6"
    return model


def vggO5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E5']), final_filter=1024, linear_layer=512, **kwargs)
    model.name = "VGGo5"
    return model


def vggO4(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E4']), **kwargs)
    model.name = "VGGo4"
    return model


def vggO3(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E3']), **kwargs)
    model.name = "VGGo3"
    return model


def vggO2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E2']), **kwargs)
    model.name = "VGGo2"
    return model


def vggO2b(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E2']), **kwargs)
    model.name = "VGGo2"
    return model


def make_layers(cfg, batch_norm=True, k_size=3, in_channels=3, pca=PCA, thresh=.999, centering=True, downsampling=None,
                dilation=1):
    layers = []
    effective_kernel_size = k_size + (k_size - 1) * (dilation - 1)
    padding = effective_kernel_size - 2
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif isinstance(v, list) or isinstance(v, tuple):
            if len(v) == 2:
                filters, stride = v
                tmp_ksize = k_size
            else:
                filters, stride, tmp_ksize = v
            tmp_padding = tmp_ksize - 2
            conv2d = nn.Conv2d(in_channels, filters, kernel_size=tmp_ksize, padding=tmp_padding, dilation=dilation,
                               stride=stride)
            if batch_norm and pca:
                layers += [conv2d,
                           Conv2DPCALayer(in_filters=filters, threshold=thresh, centering=centering, downsampling=True),
                           nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
            elif batch_norm:
                layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = filters
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=k_size, padding=padding, dilation=dilation)
            if batch_norm and pca:
                layers += [conv2d,
                           Conv2DPCALayer(in_filters=v, threshold=thresh, centering=centering, downsampling=True),
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True,
                 final_filter: int = 512, linear_layer=None, pretrained=False,
                 input_size=(32, 32), pool_size=1, regress=False, add_pca_layers=PCA, thresh=.99, centering=False,
                 dense_classifier: bool = False):
        super(VGG, self).__init__()
        self.dense_classifier = dense_classifier
        if regress:
            self.scale_factor = num_classes
            num_classes = 1
        self.regress = regress
        self.centering = centering
        self.add_pca_layers = add_pca_layers
        if linear_layer is None:
            linear_layer = final_filter // 2
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(pool_size)
        if add_pca_layers:
            if not dense_classifier:
                self.classifier = nn.Sequential(
                    nn.BatchNorm1d(final_filter * (pool_size ** 2)),
                    nn.Dropout(0.25),
                    nn.Linear(final_filter * (pool_size ** 2), linear_layer),
                    LinearPCALayer(linear_layer, thresh, centering=self.centering),
                    nn.ReLU(True),
                    nn.BatchNorm1d(linear_layer),
                    nn.Dropout(0.25),
                    nn.Linear(linear_layer, num_classes)
                )
            else:
                self.classifier = nn.Sequential(
                    nn.BatchNorm1d(final_filter * 4),
                    nn.Dropout(0.25),
                    nn.Linear(final_filter * 4, linear_layer),
                    LinearPCALayer(linear_layer, thresh, centering=self.centering),
                    nn.ReLU(True),
                    nn.BatchNorm1d(linear_layer),
                    nn.Dropout(0.25),
                    nn.Linear(linear_layer, num_classes)
                )
        else:
            if not dense_classifier:
                self.classifier = nn.Sequential(
                    nn.BatchNorm1d(final_filter * (pool_size ** 2)),
                    nn.Dropout(0.25),
                    nn.Linear(final_filter * (pool_size ** 2), linear_layer),
                    nn.ReLU(True),
                    nn.BatchNorm1d(linear_layer),
                    nn.Dropout(0.25),
                    nn.Linear(linear_layer, num_classes)
                )
            else:
                self.classifier = nn.Sequential(
                    nn.BatchNorm1d(final_filter * 4),
                    nn.Dropout(0.25),
                    nn.Linear(final_filter * 4, linear_layer),
                    nn.ReLU(True),
                    nn.BatchNorm1d(linear_layer),
                    nn.Dropout(0.25),
                    nn.Linear(linear_layer, num_classes)
                )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        if not self.dense_classifier:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.regress:
            x = x * self.scale_factor
            x = torch.round(x)
            x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg16_d1(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], dilation=1), **kwargs)
    model.name = "VGG16_D1"
    return model


def vgg16_d2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], dilation=2), **kwargs)
    model.name = "VGG16_D2"
    return model


def vgg16_d4(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], dilation=4), **kwargs)
    model.name = "VGG16_D4"
    return model


def vgg16(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    model.name = "VGG16"
    return model


def vgg16_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DS']), final_filter=256, **kwargs)
    model.name = "VGG16_S"
    return model


def vgg16_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DXS']), final_filter=128, **kwargs)
    model.name = "VGG16_XS"
    return model


def vgg16_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DXXS']), final_filter=64, **kwargs)
    model.name = "VGG16_XXS"
    return model


def vgg16_XXXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DXXXS']), final_filter=32, **kwargs)
    model.name = "VGG16_XXXS"
    return model


def vgg16_5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], k_size=5), **kwargs)
    model.name = "VGG16_5"
    return model


def vgg16_7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], k_size=7), **kwargs)
    model.name = "VGG16_7"
    return model


def vgg16_9(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], k_size=9), **kwargs)
    model.name = "VGG16_9"
    return model


def vgg16_11(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], k_size=9), **kwargs)
    model.name = "VGG16_11"
    return model


def vgg5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['V']), final_filter=64, **kwargs)
    model.name = "VGG5"
    return model


def vgg5_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['VS']), final_filter=32, **kwargs)
    model.name = "VGG5_S"
    return model


def vgg6(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['W']), final_filter=128, **kwargs)
    model.name = "VGG6"
    return model


def vgg6_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['WS']), final_filter=64, **kwargs)
    model.name = "VGG6_S"
    return model


def vgg7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['X']), final_filter=256, **kwargs)
    model.name = "VGG7"
    return model


def vgg7_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['XS']), final_filter=128, **kwargs)
    model.name = "VGG7_S"
    return model


def vgg8(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['Y']), **kwargs)
    model.name = "VGG8"
    return model


def vgg8_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['YS']), final_filter=256, **kwargs)
    model.name = "VGG8_S"
    return model


def vgg8_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['YXS']), final_filter=128, **kwargs)
    model.name = "VGG8_XS"
    return model


def vgg9(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['Z']), **kwargs)
    model.name = "VGG9"
    return model


def vgg9_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ZS']), final_filter=256, **kwargs)
    model.name = "VGG9_S"
    return model


def vgg9_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ZXS']), final_filter=128, **kwargs)
    model.name = "VGG9_XS"
    return model


def vgg9_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ZXXS']), final_filter=64, **kwargs)
    model.name = "VGG9_XXS"
    return model


def vgg9(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['Z']), **kwargs)
    model.name = "VGG9"
    return model


def vgg11(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    model.name = "VGG11"
    return model


def vgg11_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AS']), final_filter=256, **kwargs)
    model.name = "VGG11_S"
    return model


def vgg11_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AXS']), final_filter=128, **kwargs)
    model.name = "VGG11_XS"
    return model


def vgg11_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AXXS']), final_filter=64, **kwargs)
    model.name = "VGG11_XXS"
    return model


def vgg11_XXXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AXXXS']), final_filter=32, **kwargs)
    model.name = "VGG11_XXXS"
    return model


def vgg13(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    model.name = "VGG13"
    return model


def vgg13_strides(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B_Strides']), **kwargs)
    model.name = "VGG13_Strides"
    return model


def vgg13_early(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B_Early']), final_filter=512, **kwargs)
    model.name = "VGG13_Early"
    return model


def vgg13_thin(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B_Thin']), final_filter=256, **kwargs)
    model.name = "VGG13_Thin"
    return model


def vgg13_wide(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B_Wide']), final_filter=1024, **kwargs)
    model.name = "VGG13_Wide"
    return model


def vgg13_late(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B_Late']), **kwargs)
    model.name = "VGG13_Late"
    return model


def vgg13_late_wide(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B_Late_Wide']), final_filter=2560, **kwargs)
    model.name = "VGG13_Late_Wide"
    return model


def vgg13_d2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], dilation=2), **kwargs)
    model.name = "VGG13_D2"
    return model


def vgg13_d3(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], dilation=3), **kwargs)
    model.name = "VGG13_D3"
    return model


def vgg13_d4(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], dilation=4), **kwargs)
    model.name = "VGG13_D4"
    return model


def vgg13PCA(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], pca=True), add_pca_layers=True, **kwargs)
    model.name = "VGG13PCA"
    return model


def vgg13lin_readout(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['Blin'], pca=True), add_pca_layers=True, dense_classifier=True, **kwargs)
    model.name = "VGG13Lin"
    return model


def vgg13PCA_centered(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], pca=True, centering=True), add_pca_layers=True, centering=True, **kwargs)
    model.name = "VGG13PCACentered"
    return model


def vgg13PCA98(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .98
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA98"
    return model


def vgg13PCA97(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .97
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA97"
    return model


def vgg13PCA96(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .96
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA96"
    return model


def vgg13PCA95(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .95
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA95"
    return model


def vgg13PCA94(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .94
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA94"
    return model


def vgg13PCA93(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .93
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA93"
    return model


def vgg13PCA92(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .92
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA92"
    return model


def vgg13PCA91(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .91
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA91"
    return model


def vgg13PCA90(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .9
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA90"
    return model


def vgg13PCA80(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .8
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA80"
    return model


def vgg13PCA70(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .7
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA70"
    return model


def vgg13PCA60(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .6
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PC60"
    return model


def vgg13PCA50(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .5
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA50"
    return model


def vgg13PCA40(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .4
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA40"
    return model


def vgg13PCA30(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .3
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA30"
    return model


def vgg13PCA20(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    thresh = .2
    model = VGG(make_layers(cfg['B'], pca=True, thresh=thresh), add_pca_layers=True, thresh=thresh, **kwargs)
    model.name = "VGG13PCA20"
    return model


def vgg13_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['BS']), final_filter=256, **kwargs)
    model.name = "VGG13_S"
    return model


def vgg13_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['BXS']), final_filter=128, **kwargs)
    model.name = "VGG13_XS"
    return model


def vgg13_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['BXXS']), final_filter=64, **kwargs)
    model.name = "VGG13_XXS"
    return model


def vgg13_XXXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['BXXXS']), final_filter=32, **kwargs)
    model.name = "VGG13_XXXS"
    return model


def vgg19(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    model.name = "VGG19"
    return model


def vgg19m(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], in_channels=1), **kwargs)
    model.name = "VGG19m"
    return model


def vgg19_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ES']), final_filter=256, **kwargs)
    model.name = "VGG19_S"
    return model


def vgg19_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EXS']), final_filter=128, **kwargs)
    model.name = "VGG19_XS"
    return model


def vgg19_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EXXS']), final_filter=64, **kwargs)
    model.name = "VGG19_XXS"
    return model


def vgg19_XXXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EXXXS']), final_filter=32, **kwargs)
    model.name = "VGG19_XXXS"
    return model

def vgg11_d2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], dilation=2), **kwargs)
    model.name = "VGG11_D2"
    return model


def vgg13_d2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], dilation=2), **kwargs)
    model.name = "VGG13_D2"
    return model


def vgg16_d2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], dilation=2), **kwargs)
    model.name = "VGG16_D2"
    return model


def vgg19_d2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], dilation=2), **kwargs)
    model.name = "VGG19_D2"
    return model

def vgg11_d3(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], dilation=3), **kwargs)
    model.name = "VGG11_D3"
    return model


def vgg13_d3(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], dilation=3), **kwargs)
    model.name = "VGG13_D3"
    return model


def vgg16_p(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D_P']), **kwargs)
    model.name = "VGG16_P"
    return model


def vgg16_p2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D_P2']), **kwargs)
    model.name = "VGG16_P2"
    return model


def vgg16_d3(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], dilation=3), **kwargs)
    model.name = "VGG16_D3"
    return model


def vgg19_d3(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], dilation=3), **kwargs)
    model.name = "VGG19_D3"
    return model
