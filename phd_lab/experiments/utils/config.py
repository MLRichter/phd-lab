import torch
import os
from phd_lab import models, datasets, optimizers, metrics

MODEL_REGISTRY = models
DATASET_REGISTRY = datasets
OPTIMIZER_REGISTRY = optimizers
METRICS_REGISTRY = metrics

PROBE_PERFORMANCE_SAVEFILE = "probe_performances.csv"

DEFAULT_CONFIG = {
    "model": ["resnet18"],
    "epoch": [30],
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
    "downsampling": [None],

    "cache_dir": [os.environ.get("TORCH_DATADIR", "tmp")]
}


def build_saving_structure(logs_dir: str, model_name: str, dataset_name: str, output_resolution: int, run_id: str) -> str:
    save_dir: str = os.path.join(
        logs_dir,
        model_name,
        f"{dataset_name}_{output_resolution}",
        run_id
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir
