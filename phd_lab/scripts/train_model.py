from attr import attrs, attrib
from typing import Optional, Dict, Any, List
from pathlib import Path
from phd_lab import datasets, models, optimizers, metrics
from types import ModuleType
from typing import Union
from itertools import product as cproduct
from copy import deepcopy
from ..experiments.trainer import Trainer
from ..experiments.utils.config import DEFAULT_CONFIG
from ..experiments.utils.dependency_injection import get_factory, get_model, get_dataset, get_metrics, get_optimizer

import click
import json
import numpy as np


@attrs(auto_attribs=True, frozen=True, slots=True)
class Main:
    _trainer: Trainer = attrib(init=False)
    _config: Dict[str, Any]
    _dataset_module: Union[ModuleType, str] = datasets
    _model_module: Union[ModuleType, str] = models
    _optimizer_module: Union[ModuleType, str] = optimizers
    _metrics_module: Union[ModuleType, str] = metrics

    def _save_experiment_json(self):
        pass

    def _build_iterator_config(self, config: Dict[str, Any]) -> Dict[str, List[Any]]:
        return {k: v if isinstance(v, list) else [v] for k, v in config.items()}

    def _inject_values_from_default_config(self, config: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        full_config = deepcopy(DEFAULT_CONFIG)
        full_config.update(config)
        return full_config

    def __call__(self, config_path: Path, run_id: str, device: Optional[str]):
        with config_path.open() as fp:
            config: Dict[str, Any] = json.load(fp)
        iterable_config = self._build_iterator_config(config)
        operational_config = self._inject_values_from_default_config(iterable_config)
        for exp_num, (model, epoch, batch_size,
                      dataset, resolution, optimizer,
                      logs_dir, device, conv_method,
                      delta, data_parallel, downsampling) in \
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
                    operational_config['downsampling']
                )):
            print("Running experiment", exp_num, "of", np.product([len(operational_config[key])
                                                                   for key in optimizer.keys() if key != 'metrics']))
            databundle = get_dataset(get_factory(dataset, self._dataset_module), output_resolution=resolution,
                                     batch_size=batch_size,
                                     cache_dir="tmp")
            pytorch_model = get_model(get_factory(model, self._model_module), num_classes=databundle.cardinality)
            optimizerscheduler = get_optimizer(get_factory(optimizer, self._optimizer_module), model=pytorch_model)
            metric_accumulators = get_metrics(
                [get_factory(metric, self._metrics_module) for metric in operational_config['metrics']])
            trainer = Trainer(model=pytorch_model,
                              data_bundle=databundle,
                              optimizer_bundle=optimizerscheduler,
                              run_id=run_id,
                              batch_size=batch_size,
                              epochs=epoch,
                              metrics=metric_accumulators,
                              device=device,
                              logs_dir=logs_dir,
                              delta=delta,
                              data_parallel=data_parallel,
                              downsampling=downsampling)
            trainer.train()


@click.command()
@click.option("--conifg", type=str, required=True, help="Link to the configuration json")
@click.option("--device", type=str, required=True,
              help="The device to deploy the experiment on, this argument uses pytorch codes.")
@click.option("--run-id", type=str, required=True, help="the id of the run")
@click.option("--model-registry", type=str, required=False, help="registry of the model")
@click.option("--dataset-registry", type=str, required=False, help="registry of the datasets")
@click.option("--optimizer-registry", type=str, required=False, help="registry of the optimizers")
def main(config: str, device: str, run_id: str):
    pass


if __name__ == "__main__":
    main()
