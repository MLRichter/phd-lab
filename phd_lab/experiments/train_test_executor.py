from attr import attrs
from typing import List, Optional, Union
from types import ModuleType
from .trainer import Trainer
from .domain import Metric
from .utils.dependency_injection import get_metrics, get_optimizer, get_model, get_dataset, get_factory
from .utils.post_training import Extract, Project
import json

_MODES = {
    "train": lambda x: None,
    "extract": Extract(),
    "project_pca": Project('pca'),
    "project": Project('pca'),
    "project_random": Project('random')
}


@attrs(auto_attribs=True, slots=True, frozen=True)
class TrainTestExecutor:
    mode: str = "train"

    def __call__(
            self,
            exp_num: int,
            optimizer: str,
            dataset: str,
            model: str,
            metrics: List[str],
            batch_size: int,
            epoch: int,
            device: str,
            logs_dir: str,
            delta: float,
            data_parallel: bool,
            downsampling: Optional[int],
            resolution: int,
            run_id: str,
            model_module: Union[str, ModuleType],
            dataset_module: Union[str, ModuleType],
            optimizer_module: Union[str, ModuleType],
            metric_module: Union[str, Metric]
    ) -> None:
        databundle = get_dataset(get_factory(dataset, dataset_module), output_resolution=resolution,
                                 batch_size=batch_size,
                                 cache_dir="tmp")
        pytorch_model = get_model(get_factory(model, model_module), num_classes=databundle.cardinality)
        optimizerscheduler = get_optimizer(get_factory(optimizer, optimizer_module), model=pytorch_model)
        metric_accumulators = get_metrics(
            [get_factory(metric, metric_module) for metric in metrics])
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
        with open(trainer._save_path.replace('.csv', '_config.json'), 'w') as fp:
            config = {
                "model": model,
                "epoch": epoch,
                "batch_size": batch_size,

                "dataset": dataset,
                "resolution": resolution,

                "optimizer": optimizer,
                "metrics": metrics,

                "logs_dir": logs_dir,
                "device": device,

                "conv_method": ["channelwise"],
                "delta": delta,
                "data_parallel": data_parallel,
                "downsampling": downsampling
            }
            json.dump(config, fp)
        trainer.train()
        try:
            _MODES[self.mode](trainer)
        except KeyError:
            raise ValueError(f"Illegal mode {self.mode}, legal values are {list(_MODES.keys())}")
