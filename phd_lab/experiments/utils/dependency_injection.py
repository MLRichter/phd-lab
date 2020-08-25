from types import ModuleType
from typing import Union, Callable, List
from importlib import import_module
from torch.nn.modules import Module
from ..domain import Metric, ModelFactory, DatasetFactory, OptimizerFactory, DataBundle, OptimizerSchedulerBundle
import phd_lab.experiments.utils.config

_TARGET = {
    'model': phd_lab.experiments.utils.config.MODEL_REGISTRY,
    'dataset': phd_lab.experiments.utils.config.DATASET_REGISTRY,
    'optimizer': phd_lab.experiments.utils.config.OPTIMIZER_REGISTRY,
    'metric': phd_lab.experiments.utils.config.METRICS_REGISTRY
}

def _get_registry(registry: Union[str, ModuleType]) -> ModuleType:
    if isinstance(registry, ModuleType):
        return registry
    else:
        return import_module(registry)


def add_registry(registry: Union[str, ModuleType], target: str) -> ModuleType:
    _TARGET[target] = _get_registry(registry)


def get_factory(factory_name: str, registry: Union[ModuleType, str]) -> Union[ModelFactory,
                                                                              OptimizerFactory,
                                                                              DatasetFactory, Callable[[], Metric]]:
    return getattr(_get_registry(registry), factory_name)


def get_model(factory: str, num_classes: int, **kwargs) -> Module:
    return get_factory(factory, _TARGET["model"])(num_classes=num_classes, **kwargs)


def get_dataset(factory: str, batch_size: int, output_resolution: int, cache_dir: str) -> DataBundle:
    return get_factory(factory, _TARGET["dataset"])(output_size=output_resolution, batch_size=batch_size, cache_dir=cache_dir)


def get_optimizer(factory: str, model: Module, **kwargs) -> OptimizerSchedulerBundle:
    return get_factory(factory, _TARGET["optimizer"])(model=model, **kwargs)


def get_metrics(factories: List[Callable[[], Metric]]) -> List[Metric]:
    return [factory() for factory in factories]
