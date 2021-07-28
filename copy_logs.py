from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

src_folder = Path("./logs/")
tgt_folder = Path("../justinplots/logs")

files = [file for file in src_folder.glob("**/*") if file.suffix == ".csv" or file.suffix == ".json"]
tgt_files = [tgt_folder / file.relative_to(src_folder) for file in files]

for src, tgt in tqdm(zip(files, tgt_files), "Copying", total=len(files)):
    tgt.parent.mkdir(exist_ok=True, parents=True)
    copyfile(src, tgt)