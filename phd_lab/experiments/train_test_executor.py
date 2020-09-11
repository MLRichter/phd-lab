from attr import attrs
from typing import List, Optional, Union
from types import ModuleType
from .trainer import Trainer
from .domain import Metric
from .utils.dependency_injection import get_metrics, get_optimizer, get_model, get_dataset, get_factory
from .utils.post_training import Extract, Project, ReceptiveField
import json

# The modes registered in this dictionary are valid post-training strategy.
_MODES = {
    "train": lambda x: None,
    "extract": Extract(),
    "extract_5": Extract(downsampling=5),
    "extract_4": Extract(downsampling=4),
    "extract_3": Extract(downsampling=3),
    "extract_2": Extract(downsampling=2),
    "extract_1": Extract(downsampling=1),
    "project-pca": Project('pca'),
    "project": Project('pca'),
    "project-random": Project('random'),
    "receptive-field": ReceptiveField()
}


@attrs(auto_attribs=True, slots=True, frozen=True)
class TrainTestExecutor:
    """The executor object that conducts training and post-training strategies.

    Args:
        mode:   the post training mode. Must be contained in _MODE of this file.

    """

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
        """Assemble trainer and execute training as well as post-training.

        Args:
            exp_num:                The experiment number (integer)
            optimizer:              The function name of the optimizer factory
            dataset:                The function name of the dataset factory
            model:                  The function name of the model factory
            metrics:                The list of function names of the metric factories
            batch_size:             The batch size
            epoch:                  The total number of epochs to train the model
            device:                 The compute device to train the model on, may be also a list.
            logs_dir:               The log-folder.
            delta:                  Delta-threshold for computing the saturation.
            data_parallel:          Enable multi-gpu
            downsampling:           Height and width to downsample the feature maps to. If None no dowsampling is done.
            resolution:             The resolution of the input
            run_id:                 The run-id of the run
            model_module:           The module containing the model factory
            dataset_module:         The module containing the dataset factory
            optimizer_module:       The module containing the optimizer factory
            metric_module:          The module containing the metric factory
        """
        databundle = get_dataset(dataset, output_resolution=resolution,
                                 batch_size=batch_size,
                                 cache_dir="tmp")
        pytorch_model = get_model(model, num_classes=databundle.cardinality)
        optimizerscheduler = get_optimizer(optimizer, model=pytorch_model)
        metric_accumulators = get_metrics([get_factory(metric, metric_module) for metric in metrics])
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
        with open(trainer._save_path.replace('.csv', '_config.json'), 'w', encoding="ascii") as fp:
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
