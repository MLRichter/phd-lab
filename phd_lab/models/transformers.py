import torch
from self_attention_cv import ViT, ResNet50ViT

def vit_1_128(num_classes=10, img_size=256, **kwargs):
    model = ViT(img_dim=img_size, in_channels=3, patch_dim=img_size//16, num_classes=num_classes, dim=128, blocks=1)
    model.name = "ViT_1_128"
    return model



def vit_3_128(num_classes=10, img_size=256, **kwargs):
    model = ViT(img_dim=img_size, in_channels=3, patch_dim=img_size//16, num_classes=num_classes, dim=128, blocks=3)
    model.name = "ViT_3_128"
    return model


def vit_6_128(num_classes=10, img_size=256, **kwargs):
    model = ViT(img_dim=img_size, in_channels=3, patch_dim=img_size//16, num_classes=num_classes, dim=128)
    model.name = "ViT_6_128"
    return model


def vit_6_256(num_classes=10, img_size=256, **kwargs):
    model = ViT(img_dim=img_size, in_channels=3, patch_dim=img_size//16, num_classes=num_classes, dim=256)
    model.name = "ViT_6_256"
    return model


def vit_3_256(num_classes=10, img_size=256, **kwargs):
    model = ViT(img_dim=img_size, in_channels=3, patch_dim=img_size//16, num_classes=num_classes, dim=256)
    model.name = "ViT"
    return model


