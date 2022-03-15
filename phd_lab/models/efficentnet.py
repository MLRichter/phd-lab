from torchvision.models.efficientnet import efficientnet_b0 as eff_b0


def efficientnet_b0(*args, **kwargs):
    model = eff_b0(**kwargs)
    model.name = "EfficentNetB0"
    return model