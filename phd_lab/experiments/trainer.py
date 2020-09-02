from attr import attrib, attrs
from torch.nn.modules import Module
from torch.utils.data import DataLoader
import torch
from time import time
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from delve import CheckLayerSat
from delve.writers import CSVandPlottingWriter
from .domain import DataBundle, OptimizerSchedulerBundle, Metric
from .utils.config import build_saving_structure
import os
import datetime

def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


@attrs(auto_attribs=True, slots=True)
class Trainer:
    """The trainer object handles the actual training and testing of the model.

    Args:
        model:              The PyTorch-Model
        data_bundle:        The training and test data as a DataBundle
        optimizer_bundle:   Contains the optimizer and the learning rate scheduler
        run_id:             A string identifying the specific run.
        batch_size:         The batch_size to train the model on.
        epochs:             The (total) number of epochs to train the model.
        criterion:          The optimization criterionn (loss), default is cross-entropy
        metrics:            A list of metric-object
        device:             The compute device or list of compute devices to place the model(s) on
        logs_dir:           The directory to store the results
        conv_method:        The strategy for handling convolutional layers for saturation computation
        device_sat:         The device to compute the saturation on. If None, the same device is used as for
                            the model.
        delta:              The delta threshold for computing saturation
        data_parallel:      Enable or Disable multi-GPU
        downsampling:       If None, downsampling is disabled, else the feature maps will be downsampled
                            to (downsampling x downsampling) resolution
    """

    # private internal variables
    _tracker: CheckLayerSat = attrib(init=False)
    _save_path: str = attrib(init=False)
    _initial_epoch: int = attrib(init=False)
    _trained_epochs: int = attrib(init=False)
    _experiment_done: bool = attrib(init=False)

    # General Training setup
    model: Module
    data_bundle: DataBundle
    optimizer_bundle: OptimizerSchedulerBundle
    run_id: str
    batch_size: int = 32
    epochs: int = 30
    criterion: nn.modules.loss._Loss = nn.modules.CrossEntropyLoss()
    metrics: List[Metric] = attrib(factory=list)

    # Technical Setup
    device: str = 'cpu'
    logs_dir: str = './logs'

    # delve setup
    conv_method = 'channelwise'
    device_sat: Optional[str] = None
    delta: float = 0.99
    data_parallel: bool = False
    downsampling: Optional[int] = None

    def _initialize_tracker(self):
        writer = CSVandPlottingWriter(
            self._save_path.replace('.csv', ''),
            primary_metric='test_accuracy'
        )

        self._tracker = CheckLayerSat(
            self._save_path.replace('.csv', ''),
            [writer],
            self.model,
            ignore_layer_names='convolution',
            stats=['lsat', 'idim'],
            sat_threshold=self.delta,
            verbose=False,
            conv_method=self.conv_method,
            log_interval=1,
            device=self.device_sat,
            reset_covariance=True,
            max_samples=None,
            initial_epoch=self._initial_epoch,
            interpolation_strategy='nearest' if self.downsampling is not None else None,
            interpolation_downsampling=self.downsampling
        )

    def _initialize_saving_structure(self):
        save_dir: str = build_saving_structure(
            logs_dir=self.logs_dir,
            model_name=self.model.name,
            dataset_name=self.data_bundle.dataset_name,
            output_resolution=self.data_bundle.output_resolution,
            run_id=self.run_id
        )
        self._save_path = os.path.join(
            save_dir,
            f"{self.model.name}-{self.data_bundle.dataset_name}-r{self.data_bundle.output_resolution}-bs{self.batch_size}-e{self.epochs}.csv")

    def _load_model(self):
        self.model.load_state_dict(torch.load(self._save_path.replace('.csv', '.pt'))['model_state_dict'])
        if self.data_parallel:
            # TODO: make this work with DistributedDataParallel
            self.model = nn.DataParallel(self.model)
            # from torch.nn.parallel import DistributedDataParallel
        self.model = self.model.to(self.device)

    def _load_optimizer_and_scheduler(self):
        self.optimizer_bundle.optimizer.load_state_dict(
            torch.load(self._save_path.replace('.csv', '.pt'))['optimizer']
        )
        if self.optimizer_bundle.scheduler is not None:
            self.optimizer_bundle.scheduler.load_state_dict(
                torch.load(self._save_path.replace('.csv', '.pt'))['scheduler']
        )

    def _load_initial_and_trained_epoch(self):
        self._trained_epochs = torch.load(self._save_path.replace('.csv', '.pt'))['epoch']
        self._initial_epoch = self._trained_epochs + 1

    def _check_training_done(self):
        if self._initial_epoch >= self.epochs:
            self._experiment_done = True
            print(
                f'Experiment Logs for the exact same experiment with identical run_id was detected, '
                f'training will be skipped, consider using another run_id'
            )

    def _checkpointing(self):
        self._initial_epoch = 0
        self._trained_epochs = 0
        self._experiment_done = False
        self.model = self.model.to(self.device)
        if os.path.exists(self._save_path):
            self._load_initial_and_trained_epoch()
            self._check_training_done()
            self._load_model()
            self._load_optimizer_and_scheduler()
            print(
                'Resuming existing run, starting at epoch',
                self._initial_epoch+1, 'from',
                self._save_path.replace('.csv', '.pt')
                )

    def _enable_benchmark_mode_if_cuda(self):
        if "cuda" in self.device:
            from torch.backends import cudnn
            cudnn.benchmark = True

    def __attrs_post_init__(self):
        self.device_sat = self.device if self.device_sat is None else self.device_sat
        self._enable_benchmark_mode_if_cuda()
        self._initialize_saving_structure()
        self._checkpointing()
        self._initialize_tracker()

    def _reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def _eval_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        for metric in self.metrics:
            metric.update(y_true, y_pred)

    def _print_status(self, batch: int, old_time: int, dataset: DataLoader):
        metrics = [f"{metric.name}:  {round(metric.value, 3)}" for metric in self.metrics]
        print(batch, 'of', len(dataset),
              'processing time', round(time() - old_time, 3),
              *metrics
              )

    def _print_epoch_status(self, epoch: int, old_time: int, metric_dict: Dict[str, float]):
        metrics = [f"{k}:  {round(v, 3)}" for (k, v) in metric_dict.items()]
        print(
            epoch+1, 'of', self.epochs,
            'processing time', round(time() - old_time, 3),
            *metrics
        )

    def _track_results(self, prefix: str, metric_name: str, metric_value: float) -> Tuple[str, float]:
        self._tracker.add_scalar(f"{prefix}_{metric_name}", metric_value)
        return f"{prefix}_{metric_name}", metric_value

    def _track_metrics(self, prefix: str, loss: float, total: int) -> Dict[str, float]:
        result: Dict[str, float] = dict()
        for metric in self.metrics:
            name, val = self._track_results(prefix, metric.name, metric.value)
            result[name] = val
        name, val = self._track_results(prefix, "loss", loss/total)
        result[name] = val
        return result

    def _save_checkpoint(self, train_metric: Dict[str, float], test_metric: Dict[str, float], epoch: int):
        state_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer_bundle.optimizer.state_dict(),
            'scheduler': None if self.optimizer_bundle.scheduler is None else self.optimizer_bundle.scheduler.state_dict(),
            'epoch': epoch
        }
        state_dict.update(train_metric)
        state_dict.update(test_metric)
        torch.save(state_dict, self._save_path.replace('.csv', '.pt'))

    def train(self):
        """Train the model.

        The model is trained for a total number of epochs given the number of epochs provided in the constructor.
        This includes epochs this model was trained previously.

        Returns:
            The path to the saturation ans metric logs.
        """
        if self._experiment_done:
            return
        old_time = time()
        for epoch in range(self._initial_epoch, self.epochs):
            print('Start training epoch', epoch+1)
            train_metric = self.train_epoch()
            test_metric = self.test()
            train_metric.update(test_metric)
            self._print_epoch_status(epoch=epoch, old_time=old_time, metric_dict=train_metric)
            old_time = time()

            if self.optimizer_bundle.scheduler is not None:
                self.optimizer_bundle.scheduler.step()
            self._tracker.add_saturations()
            self._save_checkpoint(train_metric=train_metric, test_metric=test_metric, epoch=epoch)
        self._tracker.close()
        return self._save_path + '.csv'

    def train_epoch(self) -> Dict[str, float]:
        """Train a single epoch.

        Returns:
            A dictionary containing all metrics computed incrementally during training.
        """
        self.model.train()
        self._reset_metrics()
        running_loss = 0
        total = 0
        old_time = time()
        for batch, data in enumerate(self.data_bundle.train_dataset):
            if batch % 10 == 0 and batch != 0:
                self._print_status(batch, old_time, self.data_bundle.train_dataset)
                old_time = time()

            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer_bundle.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            self._eval_metrics(labels, outputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer_bundle.optimizer.step()

            running_loss += loss.item()
            total += self.batch_size
        return self._track_metrics('training', running_loss, total)

    def test(self):
        """Evaluate the model on the test set.

        Returns:
            The metric computed on the test set.
        """
        self._reset_metrics()
        self.model.eval()
        total = 0
        test_loss = 0
        with torch.no_grad():
            old_time = time()
            for batch, data in enumerate(self.data_bundle.test_dataset):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                test_loss += loss.item()

                self._eval_metrics(labels, outputs)

                if batch % 10 == 0 or batch == (len(self.data_bundle.test_dataset)-1):
                    self._print_status(batch, old_time, self.data_bundle.test_dataset)
                    old_time = time()

            test_metrics = self._track_metrics('test', test_loss, total)
        return test_metrics


