from pathlib import Path
from phd_lab.experiments.main import Main
import click


@click.command()
@click.option("--config", type=str, required=True, help="Link to the configuration json")
@click.option("--device", type=str, required=True,
              help="The device to deploy the experiment on, this argument uses pytorch codes.")
@click.option("--run-id", type=str, required=True, help="the id of the run")
def main(config: str, device: str, run_id: str):
    main = Main(mode='extract')
    main(config_path=Path(config), run_id=run_id, device=device)


if __name__ == "__main__":
    main()
