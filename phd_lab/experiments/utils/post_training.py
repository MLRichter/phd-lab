from pathlib import Path
from typing import List, Union, Dict, Optional

from .gradient_collection import GradientCollector, extract_gradient_from_dataset
from .pca_layers import change_all_pca_layer_thresholds, change_all_pca_layer_thresholds_and_inject_random_directions
from time import time
from ..trainer import Trainer
from torch.nn.parallel import DataParallel
import numpy as np
import pandas as pd
import os
import json
import torch
from attr import attrs
from torch.nn.modules import Module
from .extraction import LatentRepresentationCollector, extract_from_dataset
from .receptive_field import receptive_field
from .dependency_injection import get_model
from ptflops import get_model_complexity_info
from pthflops import count_ops

_MODE = {
    'pca': change_all_pca_layer_thresholds,
    'random': change_all_pca_layer_thresholds_and_inject_random_directions
}


@attrs(auto_attribs=True, slots=True, frozen=True)
class ReceptiveField:

    def _reload_model_as_sequential(self, trainer: Trainer) -> None:
        with open(trainer._save_path.replace(".csv", "_config.json")) as fp:
            model_setup = json.load(fp)
            trainer.model = get_model(model_setup['model'], num_classes=1000, noskip=True)
            trainer.model.to(trainer.device)

    def __call__(self, trainer: Trainer):
        if "ResNet" in trainer.model.name or "MPNet" in trainer.model.name:
            print("Detected ResNet-style network, "
                  "reloading model without skip connections "
                  "for receptive field computation")
            self._reload_model_as_sequential(trainer)
        receptive_field_dict = receptive_field(trainer.model,
                                               (
                                                   3,
                                                   trainer.data_bundle.output_resolution,
                                                   trainer.data_bundle.output_resolution
                                                )
                                               )
        if not os.path.exists('./receptive_field'):
            os.makedirs('./receptive_field')
        savepath = os.path.join(
            os.path.dirname(trainer._save_path),
            f"receptive_field_{trainer.model.name }_{trainer.data_bundle.dataset_name}"
            f"_{trainer.data_bundle.output_resolution}.csv")
        pd.DataFrame.from_dict(receptive_field_dict).to_csv(savepath, sep=';')


@attrs(auto_attribs=True, slots=True, frozen=True)
class Project:

    _mode: str = 'pca'
    _deltas: List[float] = [
        0.9, 0.91, 0.92, 0.93, 0.94,
        0.95, 0.96, 0.97, 0.98, 0.99,
        0.992, 0.994, 0.996, 0.998, 0.999,
        0.999, 0.9991, 0.9992, 0.9993, 0.9994,
        0.9995, 0.9996, 0.9997, 0.9998, 0.9999, 3.0
    ]

    def _save(self, trainer,
              datasets_csv: List[str],
              losses: List[float],
              model_names: List[str],
              accs: List[float],
              inference_thresholds: List[float],
              dims: List[float],
              fdims: List[float],
              sat_avg: List[float],
              downsamplings: List[int]):

        savepath = os.path.join(
            trainer.logs_dir,
            f"projections_{trainer.run_id}.csv")

        pd.DataFrame.from_dict({
            'dataset': datasets_csv,
            'loss': losses,
            'model': model_names,
            'accs': accs,
            'thresh': inference_thresholds,
            'intrinsic_dimensions': dims,
            'featurespace_dimension': fdims,
            'sat_avg': sat_avg,
            'downsampling': downsamplings
        }).to_csv(savepath, sep=';', mode='a' if os.path.exists(savepath) else 'w', header=not os.path.exists(savepath))

    def _update(self, model: Module, model_names: List[str], accs: List[float],
                acc: float, losses: List[float], loss: float, dims: List[int], indims: int,
                inference_thresholds: List[float], eval_thresh: float, fdims: List[int], fsdims: int,
                sat: List[float], sat_avg: List[float], datasets_csv: List[str],
                trainer: Trainer, downsamplings: List[int], lnames: List[str]) -> Dict[str, Union[float, int, str]]:
        model_names.append(model.name)
        accs.append(acc), losses.append(loss), dims.append(sum(indims)), inference_thresholds.append(eval_thresh)
        fdims.append(sum(fsdims))
        sats_l = ({name: [lsat] for name, lsat in zip(lnames, sat)})
        avg = np.mean(sat)
        sat_avg.append(avg)
        datasets_csv.append(f"{trainer.data_bundle.dataset_name}_{trainer.data_bundle.output_resolution}")
        downsamplings.append(trainer.downsampling)
        sats_l['avg_sat'] = avg
        sats_l['loss'] = loss
        return sats_l

    def __call__(self, trainer: Trainer):
        model = trainer.model
        model_names, accs, losses, \
        dims, inference_thresholds, fdims, sat_avg, \
        datasets_csv, downsamplings = [], [], [], [], [], [], [], [], []
        start = time()
        for eval_thresh in self._deltas:
            # change_all_pca_layer_thresholds_and_inject_random_directions(eval_thresh, model, verbose=False)
            sat, indims, fsdims, lnames = _MODE[self._mode](eval_thresh, network=model)
            print('Changed model threshold to', eval_thresh)
            result_dict = trainer.test()
            acc, loss = result_dict['test_accuracy'], result_dict['test_loss']
            print('InDims:', sum(indims), 'Acc:', acc, 'Loss:', loss, 'for', model.name, 'at threshold:', eval_thresh)
            sats_l = self._update(model=model, model_names=model_names, accs=accs, acc=acc,
                                  losses=losses, loss=loss, dims=dims, indims=indims, inference_thresholds=inference_thresholds,
                                  eval_thresh=eval_thresh, fdims=fdims, fsdims=fsdims, sat=sat, sat_avg=sat_avg,
                                  datasets_csv=datasets_csv, trainer=trainer, downsamplings=downsamplings, lnames=lnames)
            pd.DataFrame.from_dict(
                sats_l
            ).to_csv(f'{trainer._save_path.replace(".csv", "_satsl_{}.csv".format(eval_thresh))}', sep=';')
        end = time()
        print('Took:', end - start)
        self._save(trainer=trainer, datasets_csv=datasets_csv, losses=losses, model_names=model_names,
                   accs=accs, inference_thresholds=inference_thresholds, dims=dims, fdims=fdims, sat_avg=sat_avg,
                   downsamplings=downsamplings)


@attrs(auto_attribs=True, slots=True, frozen=True)
class Extract:

    latent_representation_logs: str = './latent_datasets/'
    downsampling: Optional[int] = 4
    save_feature_map_positions_individually: bool = False

    def __call__(self, trainer: Trainer):
        trainer._tracker.stop()
        model = trainer.model if not isinstance(trainer.model, DataParallel) else trainer.model.module
        print('Initializing logger')
        logger = LatentRepresentationCollector(
            model,
            savepath=os.path.join(
                self.latent_representation_logs,
                '{}_{}_{}'.format(
                    model.module.name if isinstance(model, DataParallel) else model.name,
                    trainer.data_bundle.dataset_name,
                    trainer.data_bundle.output_resolution
                )
            ),
            downsampling=self.downsampling,
            save_per_position=self.save_feature_map_positions_individually
        )
        print('Extracting training')
        model.train()
        extract_from_dataset(logger, True, model, trainer.data_bundle.train_dataset, trainer.device)
        logger.save(os.path.dirname(trainer._save_path))
        print('Extracting test')
        model.eval()
        extract_from_dataset(logger, False, model, trainer.data_bundle.test_dataset, trainer.device)
        logger.save(os.path.dirname(trainer._save_path))


@attrs(auto_attribs=True, slots=True, frozen=True)
class ComputeFLOPS:

    @staticmethod
    def _flops_to_string(flops: int, units: str = 'Mac', unit_scale: str = "G", precision: int = 2):
        if units is None:
            if flops // 10 ** 9 > 0:
                return str(round(flops / 10. ** 9, precision)) + ' GMac'
            elif flops // 10 ** 6 > 0:
                return str(round(flops / 10. ** 6, precision)) + ' MMac'
            elif flops // 10 ** 3 > 0:
                return str(round(flops / 10. ** 3, precision)) + ' KMac'
            else:
                return str(flops) + ' Mac'
        else:
            if unit_scale == 'G':
                return str(round(flops / 10. ** 9, precision)) + ' ' + unit_scale + units
            elif unit_scale == 'M':
                return str(round(flops / 10. ** 6, precision)) + ' ' + unit_scale + units
            elif unit_scale == 'K':
                return str(round(flops / 10. ** 3, precision)) + ' ' + unit_scale + units
            else:
                return str(flops) + ' ' + units

    def __call__(self, trainer: Trainer):
        resolution = 3, trainer.data_bundle.output_resolution, trainer.data_bundle.output_resolution
        fake_input = torch.rand(1, *resolution).to(trainer.device)
        ops, _ = count_ops(trainer.model, fake_input)
        macs, params = get_model_complexity_info(trainer.model, resolution, as_strings=False,
                                                 print_per_layer_stat=True, verbose=True)

        n_samples = len(trainer.data_bundle.train_dataset) * trainer.batch_size
        total_train_flops = macs * 2 * 3 * n_samples * trainer.epochs
        results = {
            "flops": ops,
            "macs": macs,
            "params": params,
            "total flops": total_train_flops,

            "str flops": self._flops_to_string(ops, "FLOPS"),
            "str macs": self._flops_to_string(macs, "MAC"),
            "str params": self._flops_to_string(params, "Params"),
            "str total flops": self._flops_to_string(total_train_flops, "FLOPS")
        }
        savefile = os.path.join(os.path.dirname(trainer._save_path), "computational_info.json")

        with open(savefile, "w+") as fp:
            json.dump(results, fp)


@attrs(auto_attribs=True, slots=True, frozen=True)
class ComputeAndPlotGradientSizes:

    @staticmethod
    def plot_grad_flow_v2(named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        from matplotlib import pyplot as plt
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        from matplotlib.lines import Line2D
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    def __call__(self, trainer: Trainer):
        trainer._tracker.stop()
        model = trainer.model if not isinstance(trainer.model, DataParallel) else trainer.model.module
        print('Initializing logger')
        logger = GradientCollector(
            model,
            savepath=os.path.dirname(trainer._save_path)
            )
        print('Extracting training')
        model.train()
        extract_gradient_from_dataset(logger, model=model, dataset=trainer.data_bundle.train_dataset, device=trainer.device, criterion=trainer.criterion, optimizer=trainer.optimizer_bundle.optimizer)
