import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, nosqueeze=False, multipath=True, res=False, double_conv=False):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.res = res
        self.nosqueeze = nosqueeze
        self.multipath = multipath
        if nosqueeze:
            squeeze_planes = inplanes
        if not nosqueeze:
            self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
            self.squeeze_activation = nn.ReLU(inplace=True)
        if multipath:
            self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                       kernel_size=1)
            self.expand1x1_activation = nn.ReLU(inplace=True)
        expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        expand3x3_activation = nn.ReLU(inplace=True)
        self.double_conv = double_conv
        primary_path = [expand3x3, expand3x3_activation]
        if self.double_conv:
            expand3x3_2 = nn.Conv2d(expand3x3_planes, expand3x3_planes, kernel_size=3, padding=1)
            expand3x3_2_activation = nn.ReLU(inplace=True)
            primary_path.append(expand3x3_2)
            primary_path.append(expand3x3_2_activation)
        self.main_path = nn.Sequential(*primary_path)

    def forward(self, x):
        if not self.nosqueeze:
            x = self.squeeze_activation(self.squeeze(x))
        if self.multipath:
            return torch.cat([
                self.expand1x1_activation(self.expand1x1(x)),
                self.main_path(x)
            ], 1)
        elif self.res:
            return torch.cat([
                x,
                self.main_path(x)
            ], 1)
        else:
            if __name__ == '__main__':
                return self
            return self.main_path(x)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', num_classes=1000, max_pooling_kernel=3):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # alibi squeeze with no dimenssion reduction
        elif version == '1_1as':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(64, 64, 64, 64),
                Fire(128, 64, 64, 64),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(128, 128, 128, 128),
                Fire(256, 128, 128, 128),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(256, 192, 192, 192),
                Fire(384, 192, 192, 192),
                Fire(384, 256, 256, 256),
                Fire(512, 256, 256, 256),
            )
        # no squeeze layers
        elif version == '1_1ns':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64, nosqueeze=True),
                Fire(128, 16, 64, 64, nosqueeze=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128, nosqueeze=True),
                Fire(256, 32, 128, 128, nosqueeze=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192, nosqueeze=True),
                Fire(384, 48, 192, 192, nosqueeze=True),
                Fire(384, 64, 256, 256, nosqueeze=True),
                Fire(512, 64, 256, 256, nosqueeze=True),
            )
            # no squeeze layers
        elif version == '1_1res':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64, multipath=False, res=True),
                Fire(64+16, 16, 64, 64, multipath=False, res=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(64+16, 32, 128, 128, multipath=False, res=True),
                Fire(128+32, 32, 128, 128, multipath=False, res=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(128+32, 48, 192, 192, multipath=False, res=True),
                Fire(192+48, 48, 192, 192, multipath=False, res=True),
                Fire(192+48, 64, 256, 256, multipath=False, res=True),
                Fire(256+64, 64, 256, 256, multipath=False, res=True),
            )
        elif version == '1_1mp':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64, multipath=False),
                Fire(128 // 2, 16, 64, 64, multipath=False),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(128 // 2, 32, 128, 128, multipath=False),
                Fire(256 // 2, 32, 128, 128, multipath=False),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(256 // 2, 48, 192, 192, multipath=False),
                Fire(384 // 2, 48, 192, 192, multipath=False),
                Fire(384 // 2, 64, 256, 256, multipath=False),
                Fire(512 // 2, 64, 256, 256, multipath=False),
            )
        elif version == '1_1mpns':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64, multipath=False, nosqueeze=True),
                Fire(128 // 2, 16, 64, 64, multipath=False, nosqueeze=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(128 // 2, 32, 128, 128, multipath=False, nosqueeze=True),
                Fire(256 // 2, 32, 128, 128, multipath=False, nosqueeze=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(256 // 2, 48, 192, 192, multipath=False, nosqueeze=True),
                Fire(384 // 2, 48, 192, 192, multipath=False, nosqueeze=True),
                Fire(384 // 2, 64, 256, 256, multipath=False, nosqueeze=True),
                Fire(512 // 2, 64, 256, 256, multipath=False, nosqueeze=True),
            )
        elif version == '1_1dc':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64, double_conv=True),
                Fire(128, 16, 64, 64, double_conv=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128, double_conv=True),
                Fire(256, 32, 128, 128, double_conv=True),
                nn.MaxPool2d(kernel_size=max_pooling_kernel, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192, double_conv=True),
                Fire(384, 48, 192, 192, double_conv=True),
                Fire(384, 64, 256, 256, double_conv=True),
                Fire(512, 64, 256, 256, double_conv=True),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        units = 256 if "mp" in version else 512
        units = 320 if "res" in version else units
        final_conv = nn.Conv2d(units, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(version, pretrained, progress, **kwargs):
    model = SqueezeNet(version, **kwargs)
    return model


def squeezenet1_0(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, progress, **kwargs)


def squeezenet1_1(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, progress, **kwargs)


def squeezenet11mpns(pretrained=False, progress=True, **kwargs):
    model = _squeezenet('1_1mpns', pretrained, progress, **kwargs)
    model.name = "SqueezeNet_NoMultiPath_NoSqueeze"
    return model


def squeezenet11res(pretrained=False, progress=True, **kwargs):
    model = _squeezenet('1_1res', pretrained, progress, **kwargs)
    model.name = "SqueezeNet_Residual"
    return model


def squeezenet11mp(pretrained=False, progress=True, **kwargs):
    model = _squeezenet('1_1mp', pretrained, progress, **kwargs)
    model.name = "SqueezeNet_NoMultiPath"
    return model


def squeezenet11ns(pretrained=False, progress=True, **kwargs):
    model = _squeezenet('1_1ns', pretrained, progress, **kwargs)
    model.name = "SqueezeNet_NoSqueeze"
    return model


def squeezenet11as(pretrained=False, progress=True, **kwargs):
    model = _squeezenet('1_1as', pretrained, progress, **kwargs)
    model.name = "SqueezeNet_AlibiSqueeze"
    return model

def squeezenet11dc(pretrained=False, progress=True, **kwargs):
    model = _squeezenet('1_1dc', pretrained, progress, **kwargs)
    model.name = "SqueezeNet_DoubleConvolution"
    return model
