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

```

+-- logs
|   +-- MyModel
|   |   +-- MyDataset1_64                                               //dataset name followed by input resolution
|   |   |   +-- MyRun                                                   //id of this specific run
|   |   |   |   +--  MyModel-MyDataset1-r64-bs128-e30_config.json       //lets repeat this specific run
|   |   |   |   +--  MyModel-MyDataset1-r64-bs128-e30.csv               //saturation and metrics
|   |   |   |   +--  MyModel-MyDataset1-r64-bs128-e30.pt                //model, lr-scheduler and optimizer states
|   |   |   |   +--  projected_results.csv                              //if you projected the networks
|   |   |   |   +--  MyModel-MyDataset1-r64-bs128-e30lsat_epoch0.png    //plots of saturation and intrinsic dimensionality
|   |   |   |   +--  MyModel-MyDataset1-r64-bs128-e30lsat_epoch1.png   
|   |   |   |   +--  MyModel-MyDataset1-r64-bs128-e30lsat_epoch2.png    
|   |   |   |   +--  .                                              
|   |   |   |   +--  .                                             
|   |   |   |   +--  .                                              
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
of model training and analysis.

### Training models
There are 4 overall scripts that will conduct a training if called. It is worth noting that 
each script is calling the same Main-Functionn object in just a slightly different configuration.
They therefore share the same command line arguments and basic execution logic.
The scripts are:
+ ``train_model.py`` train models and compute saturation along the way
+ ``infer_with_altering_delta.py`` same as train.py, after trainingg concluded the model es evaluated on the test set, while changing the delta-value of all PCA layers.
+ ``extract_latent_representations.py`` extract the latent representation of the train and test set after training has concluded.
+ ``compute_receptive_field.py`` compute the receptive field after training. Non-sequential models must implemented ``noskip: bool`` argument in their consstructor in order for this to work properly.

All of these scripts have the same arguments:
+ ``--config`` path to the config.json
+ ``--device`` compute device for the model ``cuda:n`` for the nth gpu,``cpu`` for cpu
+ ``--run-id`` the id of the run, may be any string, all experiments of this config will be saved in a subfolder with this id. This is useful if you want to repeat experiments multiple times.

#### Checkpointing
All metrics and the model itself are checkpointed after each epoch and the previous weights are overwritten.
The system will automatically resume training at the end of the last finished epoch.
If one or more trainings were completed, these trainings are skipped.
Please note that post-training actions like the extractions of latent representations will still be executed.
Furthermore runs are identified by their run-id. Runs under different run-ids generally do not recognize each other, even if they
are based on the same configuration.

### Probe Classifiers and Latent Representation Extraction
Another operation that is possible with this repository is training probe classifiers on receptive fields.
Probe Classifiers are LogisticRegression models. They are trained on the output of a neural network layer using the original labels.
The performance relative to the model performance yields an intermediate solution quality for the trained model.
You can extract the latent representation by calling ``extract_latent_representations.py``.
To train models on the latent representation call ```train_probes.py```.
The script can take the following arguments:
+ ``--config`` the config the original experiments were conducted on
+ ``-f`` the root folder of the latent representation storage is by default``./latent_representation``
+ ``-mp`` the number of processes spawned by this script. By default the number of processes equal to the number of cores on your cpu. Note that the parallelization is done 
over the number of layers, therefore more processes than layers will not yield any performance benefits.


### Using consecutive script calls of scripts to split your workload
All experiments are strictly tied to the run-id and their configuration. This means that two trained models 
are considered equal if they are trained using the same configuration parameters and run-id, regardless of the called script.
There for you could for instance run: 

```python train_model.py --config ./configs/myconfig.json --device cuda:0 --run-id MyRun```

followed by 

```python compute_receptive_field.py --config ./configs/myconfig.json --device cuda:0 --run-id MyRun```

the latter script call will recognize the previously trained models and just skip to computing 
the receptive field and add the additional results to the logs.

## Adding Models / Datasets / Optimizers
You may want to add optimizer, models and datasets to this experimental setup. Basically there is a package for each of
these ingredientes:
+ ````phd_lab.datasets````
+ ````phd_lab.models````
+ ````phd_lab.optimizers````
+ ````phd_lab.metrics````

You can add datasets, model, metrics and optimizers by importing the respective factories in the ````__init__```` file of the respective
packages.
The interfaces for the respective factories are defines as protocols in ````phd_lab.experiments.domain```` or you can
simply orient yourself on the existing once in the package.
If you want to use entirely different registries for datasets, models, optimizers and metrics you can change registry 
by setting different values for:
+ ````phd_lab.experiments.utils.config.MODEL_REGISTRY````
+ ````phd_lab.experiments.utils.config.DATASET_REGISTRY````
+ ````phd_lab.experiments.utils.config.OPTIMIZER_REGISTRY````
+ ````phd_lab.experiments.utils.config.METRICS_REGISTRY````

These registries do not need to be Module or Package-Types, they merely need to have a ````__dict__```` that maps string keys
to the respective factories.
The name in the config file must allways match a factory in order to be a valid configuration.