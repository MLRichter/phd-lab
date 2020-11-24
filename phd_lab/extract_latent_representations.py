import os
from pathlib import Path
from phd_lab.experiments.main import Main
import click

if os.name == 'nt':  # running on windows:
    import win32file
    win32file._setmaxstdio(2048)


@click.command()
@click.option("--config", type=str, required=True,
              help="Link to the configuration json")
@click.option("--device", type=str, required=True,
              help="The device to deploy the experiment on, this argument uses pytorch codes.")
@click.option("--run-id", type=str, required=True,
              help="the id of the run")
@click.option("--downsample", type=str, default=4, required=False,
              help="downsample size of the feature map")
def main(config: str, device: str, run_id: str, downsample: str):
    main = Main(mode=f'extract_{downsample}')
    main(config_path=Path(config), run_id=run_id, device=device)


if __name__ == "__main__":
    main()
