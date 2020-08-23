import torchvision

PCA = False
PRETRAINED = False


def inception3(input_size=(32, 32), num_classes=10):
    model = torchvision.models.inception.Inception3(num_classes=num_classes)
    model.name = "Inception3"
    return model


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
