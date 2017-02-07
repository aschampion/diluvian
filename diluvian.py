# -*- coding: utf-8 -*-


import argparse
import inspect
import itertools
import os

import matplotlib.pyplot as plt
import neuroglancer
import numpy as np

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Convolution3D, Input, merge
from keras.layers.core import Activation
from keras.models import load_model, Model
import keras.optimizers

from config import CONFIG, Config
from third_party.multi_gpu import make_parallel
from util import extend_keras_history, get_color_shader, roundrobin
from volumes import HDF5Volume


def make_network():
    image_input = Input(shape=tuple(CONFIG.model.block_size) + (1,), dtype='float32', name='image_input')
    mask_input = Input(shape=tuple(CONFIG.model.block_size) + (1,), dtype='float32', name='mask_input')
    ffn = merge([image_input, mask_input], mode='concat')

    # Convolve and activate before beginning the skip connection modules,
    # as discussed in the Appendix of He et al 2016.
    ffn = Convolution3D(32,
                        CONFIG.network.convolution_dim[0],
                        CONFIG.network.convolution_dim[1],
                        CONFIG.network.convolution_dim[2],
                        activation='relu',
                        border_mode='same')(ffn)

    for _ in range(0, CONFIG.network.num_modules):
        ffn = add_convolution_module(ffn)

    mask_output = Convolution3D(1,
                                CONFIG.network.convolution_dim[0],
                                CONFIG.network.convolution_dim[1],
                                CONFIG.network.convolution_dim[2],
                                border_mode='same',
                                name='mask_output',
                                activation=CONFIG.network.output_activation)(ffn)
    ffn = Model(input=[image_input, mask_input], output=[mask_output])
    if CONFIG.training.num_gpus > 1:
        ffn = make_parallel(ffn, CONFIG.training.num_gpus)
    optimizer_klass = getattr(keras.optimizers, CONFIG.optimizer.klass)
    optimizer_kwargs = inspect.getargspec(optimizer_klass.__init__)[0]
    optimizer_kwargs = {k: v for k, v in CONFIG.optimizer.kwargs.iteritems() if k in optimizer_kwargs}
    optimizer = optimizer_klass(**optimizer_kwargs)
    ffn.compile(loss='binary_crossentropy',
                optimizer=optimizer)

    return ffn


def add_convolution_module(model):
    model2 = Convolution3D(32,
                           CONFIG.network.convolution_dim[0],
                           CONFIG.network.convolution_dim[1],
                           CONFIG.network.convolution_dim[2],
                           activation='relu',
                           border_mode='same')(model)
    model2 = Convolution3D(32,
                           CONFIG.network.convolution_dim[0],
                           CONFIG.network.convolution_dim[1],
                           CONFIG.network.convolution_dim[2],
                           border_mode='same')(model2)
    model = merge([model, model2], mode='sum')
    # Note that the activation here differs from He et al 2016, as that
    # activation is not on the skip connection path.
    model = Activation('relu')(model)

    return model


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


class PredictionCopy(Callback):
    def __init__(self, kludge):
        self.kludge = kludge

    def on_batch_end(self, batch, logs={}):
        if self.kludge['inputs'] and self.kludge['outputs'] is None:
            self.kludge['outputs'] = self.model.predict(self.kludge['inputs'])


def fill_region_from_model(model_file, volumes=None, bias=True, move_batch_size=1, max_moves=None):
    if volumes is None:
        raise ValueError('Volumes must be provided.')

    regions = roundrobin(*[v.region_generator(CONFIG.model.training_fov * 4) for _, v in volumes.iteritems()])

    model = load_model(model_file)

    for region in regions:
        region.bias_against_merge = bias
        region.fill(model, verbose=True, move_batch_size=move_batch_size, max_moves=max_moves)
        viewer = region.get_viewer()
        print viewer
        s = raw_input("Press Enter to continue, a to export animation, q to quit...")
        if s == 'q':
            break
        elif s == 'a':
            region_copy = region.unfilled_copy()
            ani = region_copy.fill_animation(model, 'export.mp4', verbose=True)
            s = raw_input("Press Enter when animation is complete...")


def train_network(model_file=None, model_checkpoint_file=None, volumes=None,
                  tensorboard=False, viewer=False, metric_plot=False):
    if model_file is None:
        ffn = make_network()
    else:
        ffn = load_model(model_file)

    if model_checkpoint_file is None:
        model_checkpoint_file = 'weights.hdf5'

    if volumes is None:
        raise ValueError('Volumes must be provided.')

    f_a_bins = CONFIG.training.fill_factor_bins

    num_volumes = len(volumes)
    validation_data = {k: v.simple_training_generator(
            CONFIG.model.block_size,
            CONFIG.training.batch_size,
            CONFIG.training.validation_size,
            f_a_bins=f_a_bins,
            partition=(CONFIG.training.partitions, CONFIG.training.validation_partition)) for k, v in volumes.iteritems()}
    validation_data = roundrobin(*validation_data.values())

    # Pre-train
    training_data = {k: v.simple_training_generator(
            CONFIG.model.block_size,
            CONFIG.training.batch_size,
            CONFIG.training.training_size,
            f_a_bins=f_a_bins,
            partition=(CONFIG.training.partitions, CONFIG.training.training_partition)) for k, v in volumes.iteritems()}
    training_data = roundrobin(*training_data.values())
    history = ffn.fit_generator(training_data,
            samples_per_epoch=CONFIG.training.training_size * num_volumes,
            nb_epoch=CONFIG.training.simple_train_epochs,
            validation_data=validation_data,
            nb_val_samples=CONFIG.training.validation_size * num_volumes)

    # Moving training
    kludges = {k: {'inputs': None, 'outputs': None} for k in volumes.iterkeys()}
    callbacks = [PredictionCopy(kludge) for kludge in kludges.values()]
    callbacks.append(ModelCheckpoint(model_checkpoint_file, save_best_only=True))
    callbacks.append(EarlyStopping(patience=CONFIG.training.patience))
    if tensorboard:
        callbacks.append(TensorBoard())

    training_data = {k: v.moving_training_generator(
            CONFIG.model.training_fov,
            CONFIG.training.batch_size,
            CONFIG.training.training_size,
            kludges[k],
            f_a_bins=f_a_bins,
            partition=(CONFIG.training.partitions, CONFIG.training.training_partition)) for k, v in volumes.iteritems()}
    training_data = roundrobin(*training_data.values())
    moving_history = ffn.fit_generator(training_data,
            samples_per_epoch=CONFIG.training.training_size * num_volumes,
            nb_epoch=CONFIG.training.total_epochs,
            initial_epoch=CONFIG.training.simple_train_epochs,
            max_q_size=num_volumes,
            nb_worker=1,
            callbacks=callbacks,
            validation_data=validation_data,
            nb_val_samples=CONFIG.training.validation_size * num_volumes)
    extend_keras_history(history, moving_history)

    if viewer:
        # for _ in itertools.islice(training_data, 12):
        #     continue
        dupe_data = volumes[list(volumes.keys())[0]].simple_training_generator(
                CONFIG.model.block_size,
                CONFIG.training.batch_size,
                CONFIG.training.training_size)
        viz_ex = itertools.islice(dupe_data, 1)

        for inputs, targets in viz_ex:
            viewer = neuroglancer.Viewer(voxel_size=list(CONFIG.volume.resolution))
            viewer.add(np.transpose(inputs['image_input'][0, :, :, :, 0]),
                       name='Image')
            viewer.add(np.transpose(inputs['mask_input'][0, :, :, :, 0]),
                       name='Mask Input',
                       shader=get_color_shader(2))
            viewer.add(np.transpose(targets[0][0, :, :, :, 0]),
                       name='Mask Target',
                       shader=get_color_shader(0))
            output = ffn.predict(inputs)
            viewer.add(np.transpose(output[0, :, :, :, 0]),
                       name='Mask Output',
                       shader=get_color_shader(1))
            print viewer

            raw_input("Press any key to exit...")

    if metric_plot:
        plot_history(history)

    return history


def cli():
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

    args = parser.parse_args()

    if args.config_files:
        CONFIG = Config.from_toml(*args.config_files)
    if args.volume_file:
        volumes = HDF5Volume.from_toml(args.volume_file)
    else:
        volumes = HDF5Volume.from_toml(os.path.join(os.path.dirname(__file__), 'conf', 'cremi_datasets.toml'))

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
                               max_moves=args.max_moves)


if __name__ == "__main__":
    cli()
