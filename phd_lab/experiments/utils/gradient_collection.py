import pandas
from torch.nn import Module
import torch
import os
from os import makedirs
from os.path import exists, join
from torch.nn import Conv2d, Linear, LSTM
from torch.utils.data import DataLoader
from typing import Union, Dict, List
from shutil import rmtree
from itertools import product
import numpy as np
from time import time
import pickle

from tqdm import tqdm

if os.name == 'nt':  # running on windows:
    import win32file
    win32file._setmaxstdio(2048)


class GradientCollector:
    """This Object collects the gradients from all layers.
    """

    def __init__(self, model: Module,
                 savepath: str,
                 downsampling: int = None):
        """

        Args:
            model:              this is a pyTorch-Module
            downsampling:       downsample the latent representation to a height and width equal to the downsampling value.
            save_per_position:  saves a dataset per layer per position of the feature map instead of saving the feature maps downsamples as a whole.
        """
        self.savepath = savepath
        self.downsampling = downsampling
        self.pre_exists = False
        self.layers = self.get_layers_recursive(model)
        for name, layer in self.layers.items():
            if isinstance(layer, Conv2d) or isinstance(layer, Linear) \
                                  or isinstance(layer, LSTM):
                self._register_hooks(layer=layer,
                                     layer_name=name,
                                     interval=1)
        self.logs = {}

    def _record_stat(self, grad_output, layer_name):
        """This function is called in the backward-hook to all convolutional and linear layers.

        Args:
            grad_input:     gradient input
            grad_output:    gradient output

        Returns:
            Returns nothing, this hook is side-effect free
        """
        gradient_acc_mean = grad_output[0]
        gradient_acc_norm = gradient_acc_mean.norm()
        self.logs[layer_name] = gradient_acc_norm

    def _register_hooks(self, layer: Module, layer_name: str, interval: int) -> None:
        """Register a forward hook on a given layer.

        Args:
            layer:          the module.
            layer_name:     name of the layer.
            interval:       unused variable, needed for compatibility.
        """
        layer.name = layer_name

        def record_layer_history(layer: torch.nn.Module, grad_input, grad_output):
            """Hook to register in `layer` module."""
            #print('Inside ' + layer.name)
            #print('Inside class:' + self.__class__.__name__)
            #print('')
            #print('grad_input: ', type(grad_input))
            #print('grad_input[0]: ', type(grad_input[0]))
            #print('grad_output: ', type(grad_output))
            #print('grad_output[0]: ', type(grad_output[0]))
            #print('')
            #print('grad_input size:', grad_input[0].size())
            #print('grad_output size:', grad_output[0].size())
            #print('grad_input norm:', grad_input[0].norm())

            #activations_batch = output
            out_norm = grad_output[0].data.detach().norm().cpu().item()
            in_norm = grad_input[0].data.detach().norm().cpu().item()
            if layer.name+"-input" not in self.logs:
                self.logs[layer.name+"-input"] = in_norm
            else:
                self.logs[layer.name+"-input"] += in_norm

            if layer.name+"-output" not in self.logs:
                self.logs[layer.name+"-output"] = out_norm
            else:
                self.logs[layer.name+"-output"] += out_norm


        layer.register_backward_hook(record_layer_history)

    def get_layer_from_submodule(self, submodule: torch.nn.Module,
                                 layers: dict, name_prefix: str = '') -> Dict[str, Module]:
        """Finds all linear and convolutional layers in a network structure.

        The algorithm is recursive.

        Args:
            submodule:      the current submodule.
            layers:         the dictionary containing all layers found so far.
            name_prefix:    the prefix of the layers name. The prefix resembled the position in
                            the networks structure.

        Returns:
            the layers stored in a dictionary.
        """
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

    def get_layers_recursive(self, modules: Union[List[torch.nn.Module], torch.nn.Module]) -> Dict[str, Module]:
        """Recursive search algorithm for finding convolutional an linear layers

        Args:
            modules: maybe a single (sub)-module or a List of modules

        Returns:
            a dictionary that maps layer names to modules
        """
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

    def save(self, n_batches: int) -> None:
        """Saving the models latent representations.

        Args:
            n_batches:     number of batches

        """
        intermediate = {"name": [], "grad_in": [], "grad_out": []}
        for layer in self.logs.keys():
            name = layer.split("-input")[0].split("-output")[0]
            grad_in, grad_out = self.logs[name + "-input"] / n_batches, self.logs[name + "-output"] / n_batches
            intermediate["name"].append(name)
            intermediate["grad_in"].append(grad_in)
            intermediate["grad_out"].append(grad_out)

        pandas.DataFrame.from_dict(intermediate).drop_duplicates().to_csv(join(self.savepath, "gradient_norms.csv"))


def extract_gradient_from_dataset(logger: GradientCollector, model: Module,
                                  dataset: DataLoader, device: str, criterion, optimizer) -> None:
    """Extract latent representations from a given classification dataset.

    Args:
        logger:     The logger that collects the latent representations.
        train:      Marks the subset as training or evalutation dataset
        model:      The model from which the latent representations need to be collected.
        dataset:    The dataset, may be a torch data-loader
        device:     The device the model is deployed on, maybe any torch compatible key.
    """
    correct, total = 0, 0
    #acc_loss = None
    model.train()
    for batch, data in enumerate(tqdm(dataset,  "Accumulating Gradients")):
        print("ITER")
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        #FIXME: Add backward pass to properly compute gradients
        #FIXME: Add saving functionality
            optimizer.step()
        loss.backward()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()
    #acc_loss.backward()

    logger.save(n_batches=len(dataset))
    print('accuracy:', correct/total)
