from pathlib import Path
from phd_lab.experiments.main import Main
from typing import List
import click


@click.command()
@click.option("--config", type=str, required=True, help="Link to the configuration json", nargs=2)
@click.option("--device", type=str, required=True,
              help="The device to deploy the experiment on, this argument uses pytorch codes.")
@click.option("--run-id", type=str, required=True, help="the id of the run")
def main(config: List[str], device: str, run_id: str):
    for i, cfg in enumerate(config):
        print("Config", i, "of", len(config))
        main = Main(mode='train')
        main(config_path=Path(cfg), run_id=run_id, device=device)


if __name__ == "__main__":
    main()
