from typing import List
from .pca_layers import change_all_pca_layer_thresholds, change_all_pca_layer_thresholds_and_inject_random_directions
from time import time
from ..trainer import Trainer
import numpy as np
import pandas as pd
import os
from attr import attrs

_MODE = {
    'pca': change_all_pca_layer_thresholds,
    'random': change_all_pca_layer_thresholds_and_inject_random_directions
}


@attrs(auto_attribs=True, slots=True, frozen=True)
class Project:

    _mode: str = 'pca'

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

    def __call__(self, deltas: List[float], trainer: Trainer):
        model = trainer.model
        model_names, accs, losses, \
        dims, inference_thresholds, fdims, sat_avg, \
        datasets_csv, downsamplings = [], [], [], [], [], [], [], [], []
        start = time()
        for eval_thresh in deltas:
            # change_all_pca_layer_thresholds_and_inject_random_directions(eval_thresh, model, verbose=False)
            sat, indims, fsdims, lnames = _MODE[self.mode](eval_thresh, network=model)
            print('Changed model threshold to', eval_thresh)
            acc, loss = trainer.test(False)
            print('InDims:', sum(indims), 'Acc:', acc, 'Loss:', loss, 'for', model.name, 'at threshold:', eval_thresh)

            model_names.append(model.name)
            accs.append(acc), losses.append(loss), dims.append(sum(indims)), inference_thresholds.append(eval_thresh)
            fdims.append(sum(fsdims))
            sats_l = ({name: [lsat] for name, lsat in zip(lnames, sat)})
            avg = np.mean(sat), sat_avg.append(avg)
            datasets_csv.append(f"{trainer.data_bundle.dataset_name}_{trainer.data_bundle.output_resolution}")
            downsamplings.append(trainer.downsampling)
            sats_l['avg_sat'] = avg
            sats_l['loss'] = loss
            pd.DataFrame.from_dict(
                sats_l
            ).to_csv(f'{trainer._save_path.replace(".csv", "_satsl_{}.csv".format(eval_thresh))}', sep=';')
        end = time()
        print('Took:', end - start)
        self._save(trainer=trainer, datasets_csv=datasets_csv, losses=losses, model_names=model_names,
                   accs=accs, inference_thresholds=inference_thresholds, dims=dims, fdims=fdims, sat_avg=sat_avg,
                   downsamplings=downsamplings)
