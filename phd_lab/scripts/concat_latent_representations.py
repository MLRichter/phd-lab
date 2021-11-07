import pickle

import click
import numpy as np

from typing import Tuple
from click_pathlib import Path as cPath
from pathlib import Path

from tqdm import tqdm

from phd_lab.experiments.probe_training import load


@click.command()
@click.option("-folder", type=cPath(), help="downsamplings")
@click.option("-src", multiple=True, type=str, help="downsamplings")
@click.option("-dst", type=str, required=True)
def main(folder: Path, src: str, dst: Path):
    for mode in ["train-", "eval-"]:
        data = [load(folder / (mode + source_file)) for source_file in tqdm(src, "Loading source files")]
        all_data = np.hstack(data)
        with (folder / (mode + dst)).open("wb") as fp:
            pickle.dump(all_data, fp)


if __name__ == '__main__':
    main()
