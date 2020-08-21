from attr import attrib, attrs
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Optional, List
from delve import CheckLayerSat
from .domain import DataBundle, OptimizerSchedulerBundle, Metric


@attrs(auto_attribs=True, slots=True)
class Trainer:

    # General Training setup
    model: Module
    data_bundle: DataBundle
    optimizer_bundle: OptimizerSchedulerBundle
    batch_size: int = 32
    epochs: int = 30
    metrics: List[Metric] = attrib(factory=list)

    # Technical Setup
    device: str = 'cpu'
    logs_dir: str = 'logs'
    plot: bool = True

    # delve setup
    conv_method = 'channelwise'
    device_sat: Optional[str] = None
    delta: float = 0.99
    data_parallel: bool = False
    downsampling: Optional[int] = None






