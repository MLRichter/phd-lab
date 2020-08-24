from attr import attrs
from typing import List, Optional, Union
from types import ModuleType
from .trainer import Trainer
from .domain import Metric
from .utils.dependency_injection import get_metrics, get_optimizer, get_model, get_dataset, get_factory


def extract():
    pass


def project():
    pass


_MODES = {
    "train": lambda: None,
    "extract": extract,
    "project": project
}


@attrs(auto_attribs=True, slots=True, frozen=True)
class TrainTestExecutor:

    mode: str = "train"
    _projection_deltas: List[float] = [
        0.9, 0.91, 0.92, 0.93, 0.94,
        0.95, 0.96, 0.97, 0.98, 0.99,
        0.992, 0.994, 0.996, 0.998, 0.999,
        0.999, 0.9991, 0.9992, 0.9993, 0.9994,
        0.9995, 0.9996, 0.9997, 0.9998, 0.9999, 3.0
    ]

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
        trainer.train()
        try:
            _MODES[self.mode]()
        except KeyError:
            raise ValueError(f"Illegal mode {self.mode}, legal values are {list(_MODES.keys())}")
