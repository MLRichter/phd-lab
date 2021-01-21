from efficientnet_pytorch import EfficientNet


def efficentnet_b0(*args, **kwargs):
    model = EfficientNet.from_name('efficientnet-b0', **kwargs)
    model.name = "EfficentNet_B0"
    return model