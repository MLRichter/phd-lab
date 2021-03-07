from torchvision.models.densenet import densenet201, densenet169, densenet161, densenet121
from torchvision.models.mnasnet import mnasnet1_3
from torchvision.models.resnet import resnet152, resnet101, resnet50, resnet34, resnet18


def resnet18_pretrained(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = resnet18(pretrained=pretrained, **kwargs)
    model.name = 'ResNet18_Pretrained'
    return model


def resnet34_pretrained(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = resnet34(pretrained=pretrained, **kwargs)
    model.name = 'ResNet34_Pretrained'
    return model


def resnet50_pretrained(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = resnet50(pretrained=pretrained, **kwargs)
    model.name = 'ResNet50_Pretrained'
    return model


def resnet101_pretrained(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = resnet101(pretrained=pretrained, **kwargs)
    model.name = 'ResNet101_Pretrained'
    return model


def resnet152_pretrained(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = resnet152(pretrained=pretrained, **kwargs)
    model.name = 'ResNet152_Pretrained'
    return model


def densenet121_pretrained(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = densenet121(pretrained=pretrained, **kwargs)
    model.name = 'DenseNet121_Pretrained'
    return model


def densenet161_pretrained(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = densenet161(pretrained=pretrained, **kwargs)
    model.name = 'DenseNet161_Pretrained'
    return model


def densenet169_pretrained(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = densenet169(pretrained=pretrained, **kwargs)
    model.name = 'DenseNet169_Pretrained'
    return model


def densenet201_pretrained(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if "noskip" in kwargs:
        kwargs.pop("noskip")
    model = densenet201(pretrained=pretrained, **kwargs)
    model.name = 'DenseNet201_Pretrained'
    return model
