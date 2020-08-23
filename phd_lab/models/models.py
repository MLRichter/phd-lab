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


def Inception3(input_size=(32, 32), num_classes=10):
    model = torchvision.models.inception.Inception3(num_classes=num_classes)
    model.name = "Inception3"
    return model


import torch.nn as nn
import torch.utils.model_zoo as model_zoo


######################## POST ICML MODEL #########################

def squeezenet10(num_classes, *args, **kwargs):
    model = torchvision.models.squeezenet1_0(pretrained=PRETRAINED, num_classes=num_classes)
    model.name = 'Squeezenet10'
    return model


def squeezenet11(num_classes, *args, **kwargs):
    model = torchvision.models.squeezenet1_1(pretrained=PRETRAINED, num_classes=num_classes)
    model.name = 'Squeezenet11'
    return model


def densenet121(num_classes, *args, **kwargs):
    model = torchvision.models.densenet121(pretrained=PRETRAINED, num_classes=num_classes)
    model.name = 'DenseNet121'
    return model


def densenet169(num_classes, *args, **kwargs):
    model = torchvision.models.densenet169(pretrained=PRETRAINED, num_classes=num_classes)
    model.name = 'DenseNet169'
    return model


def densenet161(num_classes, *args, **kwargs):
    model = torchvision.models.densenet161(pretrained=PRETRAINED, num_classes=num_classes)
    model.name = 'DenseNet161'
    return model


def densenet201(num_classes, *args, **kwargs):
    model = torchvision.models.densenet201(pretrained=PRETRAINED, num_classes=num_classes)
    model.name = 'DenseNet201'
    return model


def mobilenetV2(num_classes, *args, **kwargs):
    model = torchvision.models.mobilenet_v2(pretrained=PRETRAINED, num_classes=num_classes)
    model.name = 'MobileNetV2'
    return model


def mnasnet05(num_classes, *args, **kwargs):
    model = torchvision.models.mnasnet0_5(pretrained=PRETRAINED, num_classes=num_classes)
    model.name = 'MNasNet05'
    return model


def mnasnet075(num_classes, *args, **kwargs):
    model = torchvision.models.mnasnet0_75(pretrained=PRETRAINED, num_classes=num_classes)
    model.name = 'MNasNet075'
    return model


def mnasnet10(num_classes, *args, **kwargs):
    model = torchvision.models.mnasnet1_0(pretrained=PRETRAINED, num_classes=num_classes)
    model.name = 'MNasNet10'
    return model


def mnasnet13(num_classes, *args, **kwargs):
    model = torchvision.models.mnasnet1_3(pretrained=PRETRAINED, num_classes=num_classes)
    model.name = 'MNasNet13'
    return model


def inceptionv3(num_classes, *args, **kwargs):
    model = torchvision.models.inception_v3(pretrained=PRETRAINED, num_classes=num_classes)
    model.name = 'IncetionV3'
    return model


##################################################################


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

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


def pretrainedResNet34(num_classes, *args, **kwargs):
    net = resnet34_orig(pretrained=True, num_classes=num_classes)
    net.name = 'ResNet34'
    return net


def pretrainedVGG16(num_classes, *args, **kwargs):
    net = vgg16_orig(pretrained=PRETRAINED, num_classes=num_classes)
    net.name = 'VGG16'
    return net


def pretrainedVGG19(num_classes, *args, **kwargs):
    net = vgg19_orig(pretrained=PRETRAINED, num_classes=num_classes)
    net.name = 'VGG19'
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, thresh=.99, centering=False, noskip=False):
        super(BasicBlock, self).__init__()
        self.noskip = noskip
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

        if not self.noskip:
            out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, threshold=.999, centering=False, noskip=False):
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
                 noskip=False, scale_factor=1, **kwargs):
        super(ResNet, self).__init__()
        if len(layers) <= 4:
            for _ in range(len(layers), 9):
                layers.append(None)
        self.noskip = noskip
        self.inplanes = 64 // scale_factor
        self.thresh = thresh
        self.centering = centering
        self.conv1 = nn.Conv2d(3, int(64 // scale_factor), kernel_size=7, stride=2, padding=3,
                               bias=False)
        if PCA:
            self.conv1pca = Conv2DPCALayer(64, threshold=thresh, centering=centering)
        self.bn1 = nn.BatchNorm2d(int(64 // scale_factor))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64 // scale_factor), layers[0], threshold=thresh, centering=centering)
        self.layer2 = None if layers[1] is None else self._make_layer(block, int(128 // scale_factor), layers[1],
                                                                      stride=2, threshold=thresh, centering=centering)
        self.layer3 = None if layers[2] is None else self._make_layer(block, int(256 // scale_factor), layers[2],
                                                                      stride=2, threshold=thresh, centering=centering)
        self.layer4 = None if layers[3] is None else self._make_layer(block, int(512 // scale_factor), layers[3],
                                                                      stride=2, threshold=thresh, centering=centering)
        self.layer5 = None if layers[4] is None else self._make_layer(block, int(512 // scale_factor), layers[4],
                                                                      stride=2, threshold=thresh, centering=centering)
        self.layer6 = None if layers[5] is None else self._make_layer(block, int(512 // scale_factor), layers[5],
                                                                      stride=2, threshold=thresh, centering=centering)
        self.layer7 = None if layers[6] is None else self._make_layer(block, int(512 // scale_factor), layers[6],
                                                                      stride=2, threshold=thresh, centering=centering)
        self.layer8 = None if layers[7] is None else self._make_layer(block, int(512 // scale_factor), layers[7],
                                                                      stride=2, threshold=thresh, centering=centering)
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

    def _make_layer(self, block, planes, blocks, stride=1, threshold=.999, centering=False):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion) and not self.noskip:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample if not self.noskip else None, threshold, centering,
                  noskip=self.noskip))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, noskip=self.noskip))

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


def resnet18noskip(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], noskip=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.name = 'ResNet18NoSkip'
    return model


def resnet34noskip(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], noskip=True, **kwargs)
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


class UberPaperCNN(nn.Module):
    name = 'UberNet'

    def __init__(self, input_size=(28, 28), num_classes=10, **kwargs):
        super(UberPaperCNN, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.architecture = nn.Sequential(*[
            nn.Linear(784, 200),
            LinearPCALayer(200, centering=True),
            nn.ReLU(True),
            nn.Linear(200, 200),
            LinearPCALayer(200, centering=True),
            nn.ReLU(True),
            nn.Linear(200, 10)
        ])

    def forward(self, x):
        x = x.view(-1, 28 ** 2)
        x = self.architecture(x)
        return x


def uber_net(**kwargs):
    return UberPaperCNN(**kwargs)


class LeNet(nn.Module):
    name = "LeNet"

    @staticmethod
    def _input_fc_size(input_size: int):
        conv_size1 = input_size - 2
        pool_size1 = floor((conv_size1 - 2) / 2) + 1
        conv_size2 = pool_size1 - 2
        pool_size2 = floor((conv_size2 - 2) / 2) + 1
        return pool_size2

    def __init__(self, input_size=(512, 512), num_classes=2):
        super(LeNet, self).__init__()
        self.input_size = input_size
        self.input_fc_dims = tuple(map(self._input_fc_size, input_size))
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * mul(*self.input_fc_dims), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * mul(*self.input_fc_dims))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

    'D_AP1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512, 'M'],
    'D_AP2': [64, 64, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_AP3': [64, 64, 128, 128, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_AP4': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_AP5': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 'M'],
    'D_AP6': [64, 64, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M', 512, 512, 512, 'M'],

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'E2': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'E3': [128, 128, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'E4': [256, 256, 256, 256, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'E5': [512, 256, 256, 256, 'M', 512, 256, 256, 'M', 512, 512, 1024, 'M'],
    'E6': [512, 256, 256, 256, 512, 256, 256, 512, 512, 1024],
    'E7': [512, 256, 256, 256, 512, 256, 256, 512, 512, 1024],

    'CNet0': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'CNet1': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'CNet2': [64, 64, 'M', 256, 256, 'M', 512, 512, 'M', 1024, 1024, 'M'],

    "mnist": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M'],
    "mnist2": [64, 64, 'M', 128, 128, 'M', 256, 'M'],
    "mnist3": [16, 16, 'M', 24, 24, 'M', 32, 32, 'M'],
    "mnist4": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    "mnist5b": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "mnist5": [64, 128, 'M', 512, 512, 'M', 512, 512, 'M'],

    'food1': [128, 128, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M'],
    'food2': [128, 128, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'food3': [128, 128, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'food4': [128, 128, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 512,
              'M'],

    'tiny_imgnet': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                    'M', 512, 512, 'M'],
    'tiny_imgnet2': [64, 64, 'M', 256, 256, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M'],
    'tiny_imgnet3': [64, 64, 'M', 256, 256, 'M', 256, 256, 256, 256, 'M', 512, 512, 1024, 1024, 'M'],
    'tiny_imgnet4b': [64, 64, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256, 'M', 512, 512, 'M', 1024, 1024, 'M'],
    'tiny_imgnet4': [128, 128, 256, 256, 'M', 256, 256, 256, 256, 'M', 512, 512, 1024, 1024, 'M'],
    'tiny_imgnet5': [128, 128, 256, 256, 256, 256, 256, 256, 'M', 512, 512, 1024, 1024, 'M'],
    'tiny_imgnet5b': [256, 128, 'M', 512, 256, 256, 256, 256, 256, 'M', 1024, 512, 1024, 1024, 'M'],

    'ES': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
    'EXS': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 128, 128, 128, 128, 'M'],
    'EXXS': [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M', 64, 64, 64, 64, 'M'],
    'EXXXS': [4, 4, 'M', 8, 8, 'M', 16, 16, 16, 16, 'M', 32, 32, 32, 32, 'M', 32, 32, 32, 32, 'M'],
}


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


def CNet2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['CNet2']), pool_size=1, final_filter=1024, **kwargs)
    model.name = "CNet2"
    return model


def CNet1(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['CNet1']), pool_size=1, final_filter=512, **kwargs)
    model.name = "CNet1"
    return model


def CNet1PCA(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['CNet1'], pca=True), pool_size=1, final_filter=512, add_pca_layers=True, **kwargs)
    model.name = "CNet1PCA"
    return model


def CNet0(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['CNet0']), pool_size=1, final_filter=256, **kwargs)
    model.name = "CNet0"
    return model


def ImNet5b(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['tiny_imgnet5b']), pool_size=1, final_filter=1024, **kwargs)
    model.name = "ImNet5b"
    return model


def ImNet5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['tiny_imgnet5']), pool_size=1, final_filter=1024, **kwargs)
    model.name = "ImNet5"
    return model


def ImNet4b(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['tiny_imgnet4b']), pool_size=1, final_filter=1024, **kwargs)
    model.name = "ImNet4b"
    return model


def ImNet4(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['tiny_imgnet4']), pool_size=1, final_filter=1024, **kwargs)
    model.name = "ImNet4"
    return model


def ImNet3(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['tiny_imgnet3']), pool_size=1, final_filter=1024, **kwargs)
    model.name = "ImNet3"
    return model


def ImNet2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['tiny_imgnet2']), pool_size=1, **kwargs)
    model.name = "ImNet2"
    return model


def ImNet1(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['tiny_imgnet']), pool_size=1, **kwargs)
    model.name = "ImNet1"
    return model


def FoodNet5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['food4']), pool_size=1, linear_layer=512, **kwargs)
    model.name = "FoodNet4"
    return model


def FoodNet4(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['food3']), pool_size=1, linear_layer=512, **kwargs)
    model.name = "FoodNet4"
    return model


def FoodNet3(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['food1']), pool_size=6, linear_layer=4096, **kwargs)
    model.name = "FoodNet3"
    return model


def FoodNet2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['food2']), pool_size=6, linear_layer=4096, **kwargs)
    model.name = "FoodNet2"
    return model


def FoodNet1(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['food1']), pool_size=6, linear_layer=4096, **kwargs)
    model.name = "FoodNet1"
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
            filters, stride = v
            conv2d = nn.Conv2d(in_channels, filters, kernel_size=k_size, padding=padding, dilation=dilation,
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


class TinyCAE(nn.Module):

    def __init__(self, use_pca: bool = True, thresh=.99, keepdim: bool = True, centering=True):
        super(TinyCAE, self).__init__()
        self.use_pca = use_pca
        self.centering = centering
        self.encoding_mode = False
        self.decoder = self.get_decoder()
        self.use_pca = use_pca
        self.pca_layer = LinearPCALayer(in_features=(256 ** 2) // 8, keepdim=keepdim, threshold=thresh,
                                        centering=centering)
        self.encoder = self.get_encoder()
        self._initialize_weights()

    def get_encoder(self) -> nn.Module:
        encoder = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), stride=1, padding=1, dilation=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(16, 8, (3, 3), stride=1, padding=1, dilation=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(8, 8, (3, 3), stride=1, padding=1, dilation=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), stride=2),
        )
        return encoder

    def get_decoder(self) -> nn.Module:
        decoder = nn.Sequential(
            nn.Conv2d(8, 8, (3, 3), stride=1, padding=1, dilation=1),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(8, 8, (3, 3), stride=1, padding=1, dilation=1),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(8, 16, (3, 3), stride=1, padding=1, dilation=1),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 3, (3, 3), stride=1, padding=1, dilation=1),
            nn.Sigmoid()
        )
        return decoder

    def forward(self, x):
        x = self.encoder(x)
        if self.use_pca:
            x1 = x.view(x.size(0), -1)
            x1 = self.pca_layer(x1)
            if self.pca_layer.keepdim:
                x = x1.view(x.shape)
            if self.encoding_mode:
                return x1
        x = self.decoder(x)
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


def tiny_cae(*args, **kwargs):
    model = TinyCAE(use_pca=True)
    model.name = 'TinyCAE'
    return model


class TinyCAEPCA(nn.Module):

    def __init__(self, use_pca: bool = True, thresh=.999, keepdim: bool = True, centering=True):
        self.centering = centering

        super(TinyCAEPCA, self).__init__()
        self.use_pca = use_pca
        self.thresh = thresh
        self.encoding_mode = False
        self.decoder = self.get_decoder()
        self.use_pca = use_pca
        self.pca_layer = LinearPCALayer(in_features=(256 ** 2) // 8, keepdim=keepdim, threshold=thresh)
        self.encoder = self.get_encoder()
        self._initialize_weights()

    def get_encoder(self) -> nn.Module:
        encoder = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(16, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(16, 8, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(8, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(8, 8, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(8, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), stride=2),
        )
        return encoder

    def get_decoder(self) -> nn.Module:
        decoder = nn.Sequential(
            nn.Conv2d(8, 8, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(8, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(8, 8, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(8, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(8, 16, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(16, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 3, (3, 3), stride=1, padding=1, dilation=1),
            nn.Sigmoid()
        )
        return decoder

    def forward(self, x):
        x = self.encoder(x)
        if self.use_pca:
            x1 = x.view(x.size(0), -1)
            if self.encoding_mode:
                return x1
        x = self.decoder(x)
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


def tiny_cae_pca(*args, **kwargs):
    model = TinyCAEPCA()
    model.name = 'TinyCAEPCA'
    return model


class BIGCAEPCA(nn.Module):

    def __init__(self, use_pca: bool = True, thresh=.999, keepdim: bool = True, centering=True):
        super(BIGCAEPCA, self).__init__()
        self.centering = centering
        self.use_pca = use_pca
        self.thresh = thresh
        self.encoding_mode = False
        self.decoder = self.get_decoder()
        self.use_pca = use_pca
        self.pca_layer = LinearPCALayer(in_features=(256 ** 2) // 8, keepdim=keepdim, threshold=thresh)
        self.encoder = self.get_encoder()
        self._initialize_weights()

    def get_encoder(self) -> nn.Module:
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(64, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(64, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(128, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.Conv2d(128, 128, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(128, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(256, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(256, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), stride=2),
        )
        return encoder

    def get_decoder(self) -> nn.Module:
        decoder = nn.Sequential(
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(256, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(256, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(256, 128, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(128, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.Conv2d(128, 128, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(128, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128, 64, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(64, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1, dilation=1),
            Conv2DPCALayer(64, self.thresh, centering=self.centering),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 3, (3, 3), stride=1, padding=1, dilation=1),
            nn.Sigmoid()
        )
        return decoder

    def forward(self, x):
        x = self.encoder(x)
        if self.use_pca:
            x1 = x.view(x.size(0), -1)
            if self.encoding_mode:
                return x1
        x = self.decoder(x)
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


def big_cae_pca(*args, **kwargs):
    model = BIGCAEPCA()
    model.name = 'Big_CAE_PCA'
    return model


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


def mnet5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['mnist5'], in_channels=1), final_filter=512, **kwargs)
    model.name = "mnet5"
    return model


def mnet5b(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['mnist5b'], in_channels=1), final_filter=512, **kwargs)
    model.name = "mnet5b"
    return model


def mnet4(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['mnist4'], in_channels=1), final_filter=256, **kwargs)
    model.name = "mnet4"
    return model


def mnet3(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['mnist3'], in_channels=1), final_filter=32, **kwargs)
    model.name = "mnet3"
    return model


def mnet2(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['mnist2'], in_channels=1), final_filter=256, **kwargs)
    model.name = "mnet2"
    return model


def mnet1(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['mnist'], in_channels=1), final_filter=256, **kwargs)
    model.name = "mnet1"
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
