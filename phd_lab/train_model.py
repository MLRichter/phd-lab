from pathlib import Path
from phd_lab.experiments.main import Main
from typing import List
import click


@click.command()
@click.option("--config", type=str, required=True, help="Link to the configuration json")
@click.option("--device", type=str, required=True,
              help="The device to deploy the experiment on, this argument uses pytorch codes.")
@click.option("--run-id", type=str, required=True, help="the id of the run")
@click.option("--repeat", type=int, required=False, default=1, help="number of repetitions")
def main(config: List[str], device: str, run_id: str, repeat: int):
    for i in range(repeat):
        main = Main(mode='flops')
        main(config_path=Path(config), run_id=run_id if repeat == 1 else f"{run_id}{i}", device=device)


if __name__ == "__main__":
    main()
