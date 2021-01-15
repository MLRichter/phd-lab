import os
import argparse
from multiprocessing import cpu_count
import itertools
from phd_lab.experiments.probe_training import main, PseudoArgs, parse_model
from phd_lab.experiments.utils import config as cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='folder', type=str, default="./latent_datasets", help='data folder')
    parser.add_argument('-mp', dest='mp', type=int, default=cpu_count(), help='Enable multiprocessing')
    parser.add_argument('--config', dest='config', type=str, default=None, help='Path to a config file')
    parser.add_argument('--prefix', dest='prefix', type=str, default=None, help='Postfix added to the result csv')
    parser.add_argument('--verbose', dest='verbose', type=bool, default=False, help='Show Epoch counter')
    args = parser.parse_args()
    if args.prefix is not None:
        cfg.PROBE_PERFORMANCE_SAVEFILE = args.prefix + "_" + cfg.PROBE_PERFORMANCE_SAVEFILE
    if args.config is not None:
        import json
        config = json.load(open(args.config, 'r'))
        for (model, dataset, resolution) in itertools.product(config['model'], config['dataset'], config["resolution"]):
            model_name = parse_model(model, (32, 32, 3), 10)

            pargs = PseudoArgs(model_name=model_name,
                       folder=os.path.join(args.folder, f'{model_name}_{dataset}_{resolution}'),
                       mp=args.mp)

            print(pargs)
            main(pargs)
    else:
        main(args)

