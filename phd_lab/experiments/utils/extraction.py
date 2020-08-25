from torch.nn import Module
import torch
from os import makedirs
from os.path import exists, join
from torch.nn import Conv2d, Linear, LSTM
from typing import Union
from shutil import rmtree
import numpy as np
from time import time
import pickle


class LatentRepresentationCollector:

    def __init__(self, model: Module, savepath: str, save_instantly: bool = True, downsampling: int = None):
        self.savepath = savepath
        self.downsampling = downsampling
        if exists(savepath):
            print('Found previous extraction in folder, removing it...')
            rmtree(savepath)
        makedirs(self.savepath)
        self.layers = self.get_layers_recursive(model)
        for name, layer in self.layers.items():
            if isinstance(layer, Conv2d) or isinstance(layer, Linear) \
                                  or isinstance(layer, LSTM):
                self._register_hooks(layer=layer,
                                     layer_name=name,
                                     interval=1)
        self.save_instantly = save_instantly

        self.logs = {
            'train': {},
            'eval': {}
        }
        self.record = True

    def _record_stat(self, activations_batch: torch.Tensor, layer: Module, training_state: str):
        if activations_batch.dim() == 4:  # conv layer (B x C x H x W)
            if self.downsampling is not None:
                activations_batch = torch.nn.functional.interpolate(activations_batch, self.downsampling)
            activations_batch = activations_batch.view(activations_batch.size(0), -1)
        batch = activations_batch.cpu().numpy()
        if not self.save_instantly:
            if layer.name not in self.logs[training_state]:
                self.logs[training_state][layer.name] = batch
            else:
                self.logs[training_state][layer.name] = np.vstack((self.logs[training_state][layer.name], batch))
        else:
            savepath = self.savepath+'/'+training_state+'-'+layer.name+'.p'
            if not exists(savepath):
                self.logs[training_state][layer.name] = open(savepath, 'wb')
            pickle.dump(batch, file=self.logs[training_state][layer.name])

    def _register_hooks(self, layer: Module, layer_name: str, interval: int) -> None:
        layer.name = layer_name

        def record_layer_history(layer: torch.nn.Module, input, output):
            """Hook to register in `layer` module."""

            if not self.record:
                return

            # Increment step counter
            layer.forward_iter += 1

            training_state = 'train' if layer.training else 'eval'
            activations_batch = output.data
            self._record_stat(activations_batch, layer, training_state)

        layer.register_forward_hook(record_layer_history)

    def get_layer_from_submodule(self, submodule: torch.nn.Module,
                                 layers: dict, name_prefix: str = ''):
        if len(submodule._modules) > 0:
            for idx, (name, subsubmodule) in \
                              enumerate(submodule._modules.items()):
                new_prefix = name if name_prefix == '' else name_prefix + \
                                                            '-' + name
                self.get_layer_from_submodule(subsubmodule, layers, new_prefix)
            return layers
        else:
            layer_name = name_prefix
            layer_type = layer_name
            if not isinstance(submodule, Conv2d) and not \
                   isinstance(submodule, Linear) and not \
                   isinstance(submodule, LSTM):
                print(f"Skipping {layer_type}")
                return layers
            layers[layer_name] = submodule
            print('added layer {}'.format(layer_name))
            return layers

    def get_layers_recursive(self, modules: Union[list, torch.nn.Module]):
        layers = {}
        if not isinstance(modules, list) and not hasattr(
                modules, 'out_features'):
            # is a model with layers
            # check if submodule
            submodules = modules._modules  # OrderedDict
            layers = self.get_layer_from_submodule(modules, layers, '')
        else:
            for module in modules:
                layers = self.get_layer_from_submodule(module, layers, '')
        return layers

    def save(self, model_log_path) -> None:
        with open(join(self.savepath, "model_pointer.txt"), "w+") as fp:
            fp.write(model_log_path)
        if not exists(self.savepath):
            makedirs(self.savepath)
        for mode, logs in self.logs.items():
            for layer_name, data in self.logs[mode].items():
                if isinstance(data, np.ndarray):
                    with open(self.savepath+'/'+mode+'-'+layer_name+'.p', 'wb') as p:
                        pickle.dump(data, p)
                else:
                    data.close()


def extract_from_dataset(logger: LatentRepresentationCollector, train: bool, model: Module, dataset, device: str):
    mode = 'train' if train else 'eval'
    correct, total = 0, 0
    old_time = time()
    with torch.no_grad():
        for batch, data in enumerate(dataset):
            if batch % 1 == 0 and batch != 0:
                print(batch, 'of', len(dataset), 'processing time', time() - old_time,' acc:', correct / total)
                old_time = time()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
            if 'labels' not in logger.logs[mode]:
                logger.logs[mode]['labels'] = labels.cpu().numpy()
            else:
                logger.logs[mode]['labels'] = np.hstack((logger.logs[mode]['labels'], labels.cpu().numpy()))
    print('accuracy:', correct/total)
