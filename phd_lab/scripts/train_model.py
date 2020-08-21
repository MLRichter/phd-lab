from attr import attrs, attrib
from phd_lab.experiments.trainer import Trainer
from typing import Optional, Dict, Any
from pathlib import Path
from dependency_injector.containers import DynamicContainer
from dependency_injector.providers import Factory, Callable, Singleton, Object
from phd_lab import datasets, models, optimizers, metrics
from types import ModuleType
from typing import Union
from importlib import import_module
import click


@attrs(auto_attribs=True, frozen=True, slots=True)
class Main:

    _trainer: attrib(init=False)
    _config: Dict[str, Any]
    _dataset_module: Union[ModuleType, str] = datasets
    _model_module: Union[ModuleType, str] = models
    _optimizer_module: Union[ModuleType, str] = optimizers
    _metrics_module: Union[ModuleType, str] = metrics

    def __call__(self, config_path: Path, run_id: str, device: Optional[str]):
        pass

@click.command()
@click.argument("--config", "-c", type=str)
@click.argument("--device", "-d", type=str)
@click.argument()
#TODO: Make proper click command line
def main(config: str, device: str, run_id: str):
    pass

if __name__ == "__main__":
    pass