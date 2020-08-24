from types import ModuleType
from typing import Union, Callable, List
from importlib import import_module
from torch.nn.modules import Module
from ..domain import Metric, ModelFactory, DatasetFactory, OptimizerFactory, DataBundle, OptimizerSchedulerBundle
from ..trainer import Trainer

def _get_registry(registry: Union[str, ModuleType]) -> ModuleType:
    if isinstance(registry, ModuleType):
        return registry
    else:
        return import_module(registry)


def get_factory(factory_name: str, registry: Union[ModuleType, str]) -> Union[ModelFactory,
                                                                              OptimizerFactory,
                                                                              DatasetFactory, Callable[[], Metric]]:
    return getattr(_get_registry(registry), factory_name)


def get_model(factory: ModelFactory, num_classes: int, **kwargs) -> Module:
    return factory(num_classes=num_classes, **kwargs)


def get_dataset(factory: DatasetFactory, batch_size: int, output_resolution: int, cache_dir: str) -> DataBundle:
    return factory(output_resolution=output_resolution, batch_size=batch_size, cache_dir=cache_dir)


def get_optimizer(factory: OptimizerFactory, model: Module, **kwargs) -> OptimizerSchedulerBundle:
    return factory(model=model, **kwargs)


def get_metrics(factories: List[Callable[[], Metric]]) -> List[Metric]:
    return [factory() for factory in factories]


def assemble_trainer() -> Trainer:
    return