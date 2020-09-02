# PHD-LAB Experimental Environment for Saturation based Experiments

## Introduction
The phd lab repository contains routines for training of networks, extraction of latent representations,
saturation computation and other experimental and probe training.

### Configure your Experiments
Models are configures using json-Files. The json files are collected in the ./configs
folder.
````json
{
    "model": ["resnet18", "vgg13", "myNetwork"],
    "epoch": [30],
    "batch_size": [128],

    "dataset": ["Cifar10", "ImageNet"],
    "resolution": [32, 224],

    "optimizer": ["adam", "radam"],
    "metrics": ["Accuracy", "Top5Accuracy", "MCC"],

    "logs_dir": "./logs/",
    "device": "cuda:0",

    "conv_method": ["channelwise"],
    "delta": [0.99],
    "data_parallel": false,
    "downsampling": null
}
````
Note that some elements are written as lists and some are not. 
A config can desribe an arbitrary number of experiments, where the number 
of experiments is the number of possible value combinations. The only 
exception from this rule are the metrics, which are allways provided as a list 
and are used all during every experiment.
In the above example, we train 3 models on 2 datasets using 2 optimizers. 
This result in 3x2x2=12 total experiments.
It is not necessary to set all these parameters everytime. If a parameter is not 
set a default value will be injected.
You can inspect the default value of all configuration keys in ``phd_lab.experiments.utils.config.DEFAULT_CONFIG``.

### Logging
Logging is done in a folder structure. The root folder of the logs is specified 
in ``logs_dir`` of the config file.
The system has the follow save structure

```bash

+-- logs
|   +-- MyModel
|   |   +-- MyDataset1_64                       //dataset name followed by input resolution
|   |   |   +-- MyRun                           //id of this specific run
|   |   |   |   +-- config.json                 //lets repeat this specific run
|   |   |   |   +-- saturation_results.csv
|   |   |   |   +-- saved_model.pkl             //model, lr-scheduler and optimizer states
|   |   |   |   +-- projected_results.csv       //if you projected the networks
|   |   |   |   +-- saturation_plot_epoch0.png  //plots of saturation and intrinsic dimensionality
|   +-- VGG16
|   |   +-- Cifar10_32
.   .   .   .   .     
.   .   .   .   .   
.   .   .   .   .
```

The only exception from this logging structure are the latent representation, which will be
dumped in an accordingly named folder on the top level of this repository. The reason for this is
the size of the latent representation in the hard drive. You likely want to keep your light-weight csv-results
in the logs, but may want to remove extracted latent representations on a regular basis to free up space.
(They can be reextracted from the saved model quite easily as well, so it's not even a time loss realy)

## Running Experiments
Execution of experiments is fairly straight forward. You can easily write scripts 
if you want to deviate from the out-of-the-box configurations (more on that later).
In the ``phd_lab`` folder you will find scripts handling different kinds
of model training and analysis:
+ ``train_model.py``
+ ``compute_receptive_field.py``
+ ``infer_with_altering_delta.py``
+ ``extract_latent_representations.py``
+ ``train_probes.py``

### Training models

### Projected Networks

### Probe Classifierss and Latent Representation Extraaction

## Adding Models / Datasets / Optimizers