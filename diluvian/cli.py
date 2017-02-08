# -*- coding: utf-8 -*-


import argparse
import os

from .config import CONFIG, Config
from .diluvian import fill_region_from_model, train_network
from .volumes import HDF5Volume


def main():
    global CONFIG

    common_parser = argparse.ArgumentParser(add_help=False)

    common_parser.add_argument('-c', '--config-file', action='append', dest='config_files', default=[],
                               help='Configuration files to use. For defaults, see `conf/default.toml`. ' \
                                    'Values are overwritten in the order provided.')
    common_parser.add_argument('-m', '--model-file', dest='model_file', default=None,
                               help='Existing network model file to use for prediction or continued training.')
    common_parser.add_argument('-v', '--volume-file', dest='volume_file', default=None,
                               help='Volume configuration file. For example, see `conf/cremi_datasets.toml`.')
    common_parser.add_argument('--no-in-memory', action='store_false', dest='in_memory', default=True,
                               help='Do not preload entire volumes into memory.')

    parser = argparse.ArgumentParser(description='Train or run flood-filling networks on EM data.')

    commandparsers = parser.add_subparsers(help='Commands', dest='command')

    train_parser = commandparsers.add_parser('train', parents=[common_parser],
                                             help='Train a network from labeled volumes.')
    train_parser.add_argument('-mc', '--model-checkpoint-file', dest='model_checkpoint_file', default=None,
                              help='Filename for model checkpoints. ' \
                                   'Can use Keras format arguments: https://keras.io/callbacks/#modelcheckpoint')
    train_parser.add_argument('--tensorboard', action='store_true', dest='tensorboard', default=False,
                              help='Output tensorboard log files while training.')
    train_parser.add_argument('--viewer', action='store_true', dest='viewer', default=False,
                              help='Create a neuroglancer viewer for a training sample at the end of training.')
    train_parser.add_argument('--metric-plot', action='store_true', dest='metric_plot', default=False,
                              help='Plot metric history at the end of training.')

    fill_parser = commandparsers.add_parser('fill', parents=[common_parser],
                                            help='Use a trained network to fill random regions in a volume.')
    fill_parser.add_argument('--no-bias', action='store_false', dest='bias', default=True,
                             help='Overwrite prediction mask at the end of each field of view inference ' \
                                  'rather than using the anti-merge bias update.')
    fill_parser.add_argument('--move-batch-size', dest='move_batch_size', default=1, type=int,
                             help='Maximum number of fill moves to process in each prediction batch.')
    fill_parser.add_argument('--max-moves', dest='max_moves', default=None, type=int,
                             help='Cancel filling after this many moves.')
    fill_parser.add_argument('--multi-gpu-model-kludge', dest='multi_gpu_model_kludge', default=None, type=int,
                             help='Fix using a multi-GPU trained model that was not saved properly by '
                                  'setting this to the number of training GPUs.')

    args = parser.parse_args()

    if args.config_files:
        CONFIG = Config.from_toml(*args.config_files)
    if args.volume_file:
        volumes = HDF5Volume.from_toml(args.volume_file)
    else:
        volumes = HDF5Volume.from_toml(os.path.join(os.path.dirname(__file__), '..', 'conf', 'cremi_datasets.toml'))

    if args.in_memory:
        volumes = {k: v.to_memory_volume() for k, v in volumes.iteritems()}

    if args.command == 'train':
        train_network(model_file=args.model_file,
                      model_checkpoint_file=args.model_checkpoint_file,
                      volumes=volumes,
                      tensorboard=args.tensorboard,
                      viewer=args.viewer,
                      metric_plot=args.metric_plot)
    elif args.command == 'fill':
        fill_region_from_model(args.model_file,
                               volumes=volumes,
                               bias=args.bias,
                               move_batch_size=args.move_batch_size,
                               max_moves=args.max_moves,
                               multi_gpu_model_kludge=args.multi_gpu_model_kludge)


if __name__ == "__main__":
    main()
