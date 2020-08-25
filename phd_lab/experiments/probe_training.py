import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LogisticRegressionModel
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from multiprocessing import Pool
from attr import attrs, attrib


@attrs(auto_attribs=True, slots=True)
class PseudoArgs:
    model_name: str
    folder: str
    mp: int
    save_path: str = attrib(init=False)
    overwrite: bool = False

    def __attrs_post_init__(self):
        self.save_path = open(os.path.join(self.folder, "model_pointer.txt"), "r").read()


def dataset_from_array(data: np.ndarray, targets: np.ndarray):
    tensor_x = torch.Tensor(data)  # transform to torch tensor
    tensor_y = torch.Tensor(targets).long()
    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    dataloader = DataLoader(dataset, batch_size=1000)  # create your dataloader
    return dataloader


def filter_files_by_string_key(files: List[str], key: str) -> List[str]:
    return [file for file in files if key in file]


def seperate_labels_from_data(files: List[str]) -> Tuple[List[str], List[str]]:
    data_files = [file for file in files if '-labels' not in file]
    label_file = [file for file in files if '-labels' in file]
    return data_files, label_file


def get_all_npy_files(folder: str) -> List[str]:
    all_files = os.listdir(folder)
    filtered_files = filter_files_by_string_key(all_files, '.p')
    full_paths = [os.path.join(folder, file) for file in filtered_files]
    return full_paths


def obtain_all_dataset(folder: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    all_files = get_all_npy_files(folder)
    data, labels = seperate_labels_from_data(all_files)
    train_data, train_label = filter_files_by_string_key(data, 'train-'), filter_files_by_string_key(labels, 'train-')
    eval_data, eval_label = filter_files_by_string_key(data, 'eval-'), filter_files_by_string_key(labels, 'eval-')
    train_set = [elem for elem in product(train_data, train_label)]
    eval_set = [elem for elem in product(eval_data, eval_label)]
    return train_set, eval_set


def loadall(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def load(filename: str) -> np.ndarray:
    return np.vstack([batch for batch in loadall(filename)])


def get_data_annd_labels(data_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    return load(data_path), np.squeeze(load(label_path))


def train_model(data_path: str, labels_path: str) -> LogisticRegressionModel:
    print('Loading training data from', data_path)
    data, labels = get_data_annd_labels(data_path, labels_path)
    print('Training data obtained with shape', data.shape)
    model = LogisticRegressionModel(multi_class='multinomial', n_jobs=6, solver='saga', verbose=0).fit(data, labels)#train(model, data_loader)
    return model


def obtain_accuracy(model: LogisticRegressionModel, data_path, label_path: str) -> float:
    data, labels = get_data_annd_labels(data_path, label_path)
    print('Loaded data:', data_path)
    print('Evaluating with data of shape', data.shape)
    preds = model.predict(data)
    return accuracy_score(labels, preds)


def train_model_for_data(train_set: Tuple[str, str], eval_set: Tuple[str, str]):
    print('Training model')
    model = train_model(*train_set)
    print('Obtaining metrics')
    train_acc = obtain_accuracy(model, *train_set)
    eval_acc = obtain_accuracy(model, *eval_set)
    print(os.path.basename(train_set[0]))
    print('Train acc', train_acc)
    print('Eval acc:', eval_acc)
    return train_acc, eval_acc


def train_model_for_data_mp(args):
    return train_model_for_data(*args)


def main(args: PseudoArgs):
    if os.path.exists(os.path.join(args.save_path, 'probe_performance.csv')):
        print('Detected existing results')
        if args.overwrite:
            print("overwriting is enabled. Training will continue and previous results will be overwritten.")
        else:
            print("overwriting is disabled, stopping...")
            return
    names, t_accs, e_accs = [], [], []
    train_set, eval_set = obtain_all_dataset(args.folder)
    if len(train_set) != len(eval_set):
        raise FileNotFoundError(f"Number of training sets ({len(train_set)}) does not"
                                f"match the number of test sets ({len(eval_set)})."
                                f"Make sure the datas has been extracted correctly. Consider rerunning "
                                f"extraction.")
    fargs = []
    for train_data, eval_data in zip(train_set, eval_set):
        names.append(os.path.basename(train_data[0][:-2]))
        if args.mp == 0:
            print('Multiprocessing is disabled starting training...')
            train_acc, eval_acc = train_model_for_data(train_data, eval_data)
            t_accs.append(train_acc)
            e_accs.append(eval_acc)
            pd.DataFrame.from_dict(
                {
                    'name': names,
                    'train_acc': t_accs,
                    'eval_acc': e_accs
                }
            ).to_csv(os.path.join(args.save_path, "probe_performances.csv"), sep=';')
        else:
            fargs.append((train_data, eval_data))

    if args.mp != 0:
        with Pool(args.mp) as p:
            results = p.map(train_model_for_data_mp, fargs)
        for i, result in enumerate(results):
            t_accs.append(result[0])
            e_accs.append(result[1])
        pd.DataFrame.from_dict(
            {
                'name': names,
                'train_acc': t_accs,
                'eval_acc': e_accs
            }
        ).to_csv(os.path.join(args.save_path, 'probe_performance.csv'), sep=';')


def parse_model(model_name, shape, num_classes):
    try:
        from phd_lab import models
        model = models.__dict__[model_name](input_size=shape, num_classes=num_classes)
    except KeyError:
        raise NameError("%s doesn't exist." % model_name)
    return model.name
