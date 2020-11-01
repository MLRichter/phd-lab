from pathlib import Path
import click
import json
from click_pathlib import Path as cPath
from typing import Dict, List, Tuple, Any
from shutil import copy
from tqdm import tqdm


def copy_files(src: List[Path], dst: List[Path]):
    assert len(src) == len(dst)
    for s, d in tqdm(zip(src, dst), total=len(src)):
        d.parent.mkdir(parents=True, exist_ok=True)
        copy(s, d)


def create_destination_path(train_or_val_folder: Path, raw_paths: List[Path]) -> List[Path]:
    result = []
    for path in tqdm(raw_paths):
        new_path = train_or_val_folder / (str(path.parent.parent.name) + '_' + str(path.parent.name)) / path.name
        result.append(new_path)
    return result


def obtain_source_paths(src_path: Path, dataset_dict: Dict[str, List[Dict[str, str]]]) -> Tuple[List[Path], List[Path]]:
    paths = []
    raw_paths = []
    for image_dict in tqdm(dataset_dict["images"]):
        file = src_path / image_dict["file_name"]
        assert file.exists()
        paths.append(file)
        raw_paths.append(Path(image_dict["file_name"]))
    return paths, raw_paths


def get_json(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r") as fp:
        json_file = json.load(fp)
    return json_file


@click.command()
@click.option("-src", type=cPath(), required=True)
@click.option("-dst", type=cPath(), required=True)
@click.option("--train-json", type=str, required=False, default="train2019.json")
@click.option("--val-json", type=str, required=False, default="val2019.json")
def main(src: Path, dst: Path, train_json: str, val_json: str):
    train_json_dict = get_json(src/train_json)
    val_json_dict = get_json(src/val_json)
    for (folder_name, dataset) in [("train", train_json_dict), ("test", val_json_dict)]:
        paths, raw_paths = obtain_source_paths(src, dataset)
        dest_paths = create_destination_path(train_or_val_folder=dst / folder_name, raw_paths=raw_paths)
        copy_files(paths, dest_paths)


if __name__ == "__main__":
    main()