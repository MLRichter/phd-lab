from typing import Tuple
from pathlib import Path
from phd_lab.experiments.main import Main
import click
import os
import argparse
from multiprocessing import cpu_count
import itertools
from phd_lab.experiments.probe_training import main as main_probes, PseudoArgs, parse_model
from phd_lab.experiments.utils import config as app_config



@click.command()
@click.option("--config", type=str, required=True, help="Link to the configuration json")
@click.option("--device", type=str, required=True,
              help="The device to deploy the experiment on, this argument uses pytorch codes.")
@click.option("--run-id", type=str, required=True, help="the id of the run")
@click.option("-mp", type=int, required=True, help="Number of cpu cores to use for probe training")
@click.option("--prefix", type=str, required=True, default="", help="prefix of the probe file")
@click.option("-d", multiple=True, type=int, help="downsamplings")
@click.option("--folder", type=str, default="./latent_datasets")
def main(config: str, device: str, run_id: str, mp: int, prefix: str, d: Tuple[int], folder: str):
    original_savefile = app_config.PROBE_PERFORMANCE_SAVEFILE
    for downsample in d:
        main = Main(mode=f'extract_{downsample}')
        main(config_path=Path(config), run_id=run_id, device=device)
        if prefix is not None:
            app_config.PROBE_PERFORMANCE_SAVEFILE = prefix + f"{downsample}x{downsample}" + "_" + original_savefile
        import json
        cfg = json.load(open(config, 'r'))
        for (model, dataset, resolution) in itertools.product(cfg['model'], cfg['dataset'],
                                                                  cfg["resolution"]):
            model_name = parse_model(model, (32, 32, 3), 10)

            pargs = PseudoArgs(model_name=model_name,
                                folder=os.path.join(folder, f'{model_name}_{dataset}_{resolution}'),
                                mp=mp)

            print(pargs)
            main_probes(pargs)


if __name__ == "__main__":
    main()