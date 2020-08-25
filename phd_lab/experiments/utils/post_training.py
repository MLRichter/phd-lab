from typing import List, Union, Dict
from .pca_layers import change_all_pca_layer_thresholds, change_all_pca_layer_thresholds_and_inject_random_directions
from time import time
from ..trainer import Trainer
import numpy as np
import pandas as pd
import os
import json
from attr import attrs
from torch.nn.modules import Module
from .extraction import LatentRepresentationCollector, extract_from_dataset
from .receptive_field import receptive_field
from .dependency_injection import get_model

_MODE = {
    'pca': change_all_pca_layer_thresholds,
    'random': change_all_pca_layer_thresholds_and_inject_random_directions
}


@attrs(auto_attribs=True, slots=True, frozen=True)
class ReceptiveField:

    def _reload_model_as_sequential(self, trainer: Trainer) -> None:
        with open(trainer._save_path.replace(".csv", "_config.json")) as fp:
            model_setup = json.load(fp)
            trainer.model = get_model(model_setup['model'], num_classes=10, noskip=True)
            trainer.model.to(trainer.device)

    def __call__(self, trainer: Trainer):
        if "ResNet" in trainer.model.name:
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
    downsampling: int = 4

    def __call__(self, trainer: Trainer):
        trainer._tracker.stop()
        model = trainer.model
        print('Initializing logger')
        logger = LatentRepresentationCollector(model, savepath=os.path.join(self.latent_representation_logs,
                                                                            '{}_{}_{}'.format(model.name,
                                                                                              trainer.data_bundle.dataset_name,
                                                                                              trainer.data_bundle.output_resolution)
                                                                            ), downsampling=self.downsampling)
        print('Extracting training')
        extract_from_dataset(logger, True, model, trainer.data_bundle.train_dataset, trainer.device)
        print('Extracting test')
        model.eval()
        extract_from_dataset(logger, False, model, trainer.data_bundle.test_dataset, trainer.device)
        logger.save(os.path.dirname(trainer._save_path))
