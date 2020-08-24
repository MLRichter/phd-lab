import torch

DEFAULT_CONFIG = {
    "model": ["resnet18"],
    "epoch": ["30"],
    "batch_size": [128],

    "dataset": ["Cifar10"],
    "resolution": [32],

    "optimizer": ["adam"],
    "metrics": ["Accuracy", "Top5Accuracy"],

    "logs_dir": ["./logs/"],
    "device": ['cuda:0' if torch.cuda.is_available() else 'cpu'],

    "conv_method": ["channelwise"],
    "delta": [0.99],
    "data_parallel": [False],
    "downsampling": [None]
}