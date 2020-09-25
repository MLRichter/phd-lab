from attr import attrs, attrib
from typing import Optional, Dict, Any, List
from pathlib import Path
from phd_lab import datasets, models, optimizers, metrics
from types import ModuleType
from typing import Union
from itertools import product as cproduct
from copy import deepcopy
from phd_lab.experiments.trainer import Trainer
from phd_lab.experiments.utils.config import DEFAULT_CONFIG
from phd_lab.experiments.utils.dependency_injection import add_registry
from phd_lab.experiments.train_test_executor import TrainTestExecutor

import json
import numpy as np


@attrs(auto_attribs=True, frozen=True, slots=True)
class Main:
    """The main execution function.

    This function handles training and post-training actions.
    Args:
        dataset_module:     Module containing the dataset factories
        model_module:       Module containing the model factories
        optimizer_module:   Module containing the optimizer factories
        metrics_module:     Module containing the metric factories
        mode:               String key for post-training strategy.
    """

    _trainer: Trainer = attrib(init=False)
    _dataset_module: Union[ModuleType, str] = datasets
    _model_module: Union[ModuleType, str] = models
    _optimizer_module: Union[ModuleType, str] = optimizers
    _metrics_module: Union[ModuleType, str] = metrics
    _mode: str = "extract"

    def __attrs_post_init__(self):
        add_registry(self._model_module, 'model')
        add_registry(self._dataset_module, 'dataset')
        add_registry(self._optimizer_module, 'optimizer')
        add_registry(self._metrics_module, 'metrics')

    def _build_iterator_config(self, config: Dict[str, Any]) -> Dict[str, List[Any]]:
        return {k: v if isinstance(v, list) else [v] for k, v in config.items()}

    def _inject_values_from_default_config(self, config: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        full_config = deepcopy(DEFAULT_CONFIG)
        full_config.update(config)
        return full_config

    def __call__(self, config_path: Path, run_id: str, device: Optional[str]):
        with config_path.open() as fp:
            config: Dict[str, Any] = json.load(fp)
            if not "device" in config:
                config["device"] = device
        iterable_config = self._build_iterator_config(config)
        operational_config = self._inject_values_from_default_config(iterable_config)
        for exp_num, (model, epoch, batch_size,
                      dataset, resolution, optimizer,
                      logs_dir, device, conv_method,
                      delta, data_parallel, downsampling, cache_dir) in \
                enumerate(cproduct(
                    operational_config['model'],
                    operational_config['epoch'],
                    operational_config['batch_size'],
                    operational_config['dataset'],
                    operational_config['resolution'],
                    operational_config['optimizer'],
                    operational_config['logs_dir'],
                    operational_config['device'],
                    operational_config['conv_method'],
                    operational_config['delta'],
                    operational_config['data_parallel'],
                    operational_config['downsampling'],
                    operational_config['cache_dir']
                )):
            print("Running experiment", exp_num+1, "of", np.product([len(operational_config[key])
                                                                   for key in operational_config.keys() if key != 'metrics']))
            executor = TrainTestExecutor(self._mode)
            executor(
                exp_num=exp_num,
                optimizer=optimizer,
                dataset=dataset,
                model=model,
                metrics=operational_config['metrics'],
                batch_size=batch_size,
                epoch=epoch,
                device=device,
                logs_dir=logs_dir,
                delta=delta,
                data_parallel=data_parallel,
                downsampling=downsampling,
                resolution=resolution,
                run_id=run_id,
                model_module=self._model_module,
                dataset_module=self._dataset_module,
                optimizer_module=self._optimizer_module,
                metric_module=self._metrics_module,
                cache_dir=cache_dir)
