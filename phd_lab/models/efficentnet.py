from torchvision.models.efficientnet import efficientnet_b0

def efficentnet_b0(*args, **kwargs):
    model = efficientnet_b0()
    model.name = "EfficentNet_B0"
    return model