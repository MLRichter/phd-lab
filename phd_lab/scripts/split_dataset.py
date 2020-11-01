from shutil import rmtree, copy
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import click_pathlib
import click


def searching_all_files(directory: Path) -> List[Path]:
    dirpath = Path(directory)
    assert(dirpath.is_dir())
    file_list = []
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(searching_all_files(x))
    return file_list


def setup_target_structure(dst: Path, classes: List[Path]) -> Dict[Path, Tuple[Path, Path]]:
    class_paths: Dict[Path, Path] = {cls_path: (dst / "train" / cls_path.name, dst / "test" / cls_path.name) for cls_path in classes}
    if dst.exists():
        if input(f"Directory {str(dst)} exists, do you want to overwrite it? [y/n]") == "y" and input("sure?") == "y":
            rmtree(dst)
        else:
            raise ValueError("Path allready exist, this script will not continue without explicit permission to delete")
    for _, (train_path, test_path) in class_paths.items():
        train_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)
    return class_paths


def get_classes(src: Path) -> List[Path]:
    return [x for x in src.iterdir()]


def copy_files(src: Path, files: List[Path], dst: Path) -> None:
    for d in tqdm(files):
        target = dst / d.relative_to(src).parent
        target.mkdir(exist_ok=True, parents=True)
        copy(d, target)


def copy_files_for_classes(source_folder: Path,
                           target_folder_train: Path,
                           target_folder_test: Path,
                           test_size: Path
                           ) -> None:
    print("Start copying", str(source_folder), "to", str(target_folder_train), "and",
          str(target_folder_test), "with a test pecentage of", str(test_size))
    all_files = searching_all_files(source_folder)
    print("Found", len(all_files), "files")
    train, test = train_test_split(all_files, test_size=test_size)
    print("Copying training data")
    copy_files(source_folder, train, target_folder_train)
    print("Copying testing data")
    copy_files(source_folder, test, target_folder_test)


def copy_all_files(file_mapper: Dict[Path, Path], test_size: float) -> None:
    for src, (train_target, test_target) in file_mapper.items():
        copy_files_for_classes(src, target_folder_train=train_target, target_folder_test=test_target, test_size=test_size)


@click.command()
@click.option("-src", type=click_pathlib.Path(), required=True)
@click.option("-dst", type=click_pathlib.Path(), required=True)
@click.option("--test-size", type=float, required=False, default=0.2)
def main(src: Path, dst: Path, test_size: float):
    print(src, dst, test_size)
    classes = get_classes(src)
    print("Found", len(classes), "classes:", *[cls.name for cls in classes])
    print("Setting up goal structure")
    path_mapper = setup_target_structure(dst, classes)
    copy_all_files(path_mapper, test_size)


if __name__ == "__main__":
    main()