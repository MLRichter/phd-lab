from pathlib import Path
from phd_lab.experiments.main import Main
import click


@click.command()
@click.option("--config", type=str, required=True, help="Link to the configuration json")
@click.option("--device", type=str, required=True,
              help="The device to deploy the experiment on, this argument uses pytorch codes.")
@click.option("--run-id", type=str, required=True, help="the id of the run")
@click.option("--downsample", type=int, required=False, help="downsample size of the feature map")
def main(config: str, device: str, run_id: str, downsample: int = 4):
    main = Main(mode=f'extract_{downsample}')
    main(config_path=Path(config), run_id=run_id, device=device)


if __name__ == "__main__":
    main()
