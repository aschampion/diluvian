# -*- coding: utf-8 -*-


import importlib
import inspect
import itertools

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import neuroglancer
import numpy as np

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Convolution3D, Input, merge
from keras.layers.core import Activation
from keras.models import load_model, Model
import keras.optimizers

from .config import CONFIG
from .third_party.multi_gpu import make_parallel
from .util import extend_keras_history, get_color_shader, roundrobin, write_keras_history_to_csv
from .volumes import simple_training_generator, moving_training_generator


def make_flood_fill_network():
    image_input = Input(shape=tuple(CONFIG.model.block_size) + (1,), dtype='float32', name='image_input')
    mask_input = Input(shape=tuple(CONFIG.model.block_size) + (1,), dtype='float32', name='mask_input')
    ffn = merge([image_input, mask_input], mode='concat')

    # Convolve and activate before beginning the skip connection modules,
    # as discussed in the Appendix of He et al 2016.
    ffn = Convolution3D(CONFIG.network.convolution_filters,
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

    return ffn


def add_convolution_module(model):
    model2 = Convolution3D(CONFIG.network.convolution_filters,
                           CONFIG.network.convolution_dim[0],
                           CONFIG.network.convolution_dim[1],
                           CONFIG.network.convolution_dim[2],
                           activation='relu',
                           border_mode='same')(model)
    model2 = Convolution3D(CONFIG.network.convolution_filters,
                           CONFIG.network.convolution_dim[0],
                           CONFIG.network.convolution_dim[1],
                           CONFIG.network.convolution_dim[2],
                           border_mode='same')(model2)
    model = merge([model, model2], mode='sum')
    # Note that the activation here differs from He et al 2016, as that
    # activation is not on the skip connection path. However, this is not
    # likely to be important, see:
    # http://torch.ch/blog/2016/02/04/resnets.html
    # https://github.com/gcr/torch-residual-networks
    model = Activation('relu')(model)

    return model


def compile_network(model):
    optimizer_klass = getattr(keras.optimizers, CONFIG.optimizer.klass)
    optimizer_kwargs = inspect.getargspec(optimizer_klass.__init__)[0]
    optimizer_kwargs = {k: v for k, v in CONFIG.optimizer.kwargs.iteritems() if k in optimizer_kwargs}
    optimizer = optimizer_klass(**optimizer_kwargs)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer)


def plot_history(history):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    fig.suptitle('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper right')

    return fig


class PredictionCopy(Callback):
    def __init__(self, kludge):
        self.kludge = kludge

    def on_batch_end(self, batch, logs={}):
        if self.kludge['inputs'] and self.kludge['outputs'] is None:
            self.kludge['outputs'] = self.model.predict(self.kludge['inputs'])


def fill_region_from_model(model_file, volumes=None, bias=True, move_batch_size=1,
                           max_moves=None, multi_gpu_model_kludge=None):
    if volumes is None:
        raise ValueError('Volumes must be provided.')

    regions = roundrobin(*[v.region_generator(CONFIG.model.training_fov * 4 - 3) for _, v in volumes.iteritems()])

    model = load_model(model_file)

    for region in regions:
        region.bias_against_merge = bias
        region.fill(model,
                    verbose=True,
                    move_batch_size=move_batch_size,
                    max_moves=max_moves,
                    multi_gpu_pad_kludge=multi_gpu_model_kludge)
        viewer = region.get_viewer()
        print viewer
        s = raw_input("Press Enter to continue, a to export animation, q to quit...")
        if s == 'q':
            break
        elif s == 'a':
            region_copy = region.unfilled_copy()
            # Must assign the animation to a variable so that it is not GCed.
            ani = region_copy.fill_animation(model, 'export.mp4', verbose=True) # noqa
            s = raw_input("Press Enter when animation is complete...")


def train_network(model_file=None, volumes=None,
                  model_output_filebase=None, model_checkpoint_file=None,
                  tensorboard=False, viewer=False, metric_plot=False):
    if model_file is None:
        factory_mod_name, factory_func_name = CONFIG.network.factory.rsplit('.', 1)
        factory_mod = importlib.import_module(factory_mod_name)
        factory = getattr(factory_mod, factory_func_name)
        ffn = factory()
    else:
        ffn = load_model(model_file)

    # Multi-GPU models are saved as a single-GPU model prior to compilation,
    # so if loading from such a model file it will need to be recompiled.
    if not hasattr(ffn, 'optimizer'):
        compile_network(ffn)

    if model_output_filebase is None:
        model_output_filebase = 'model_output'

    if volumes is None:
        raise ValueError('Volumes must be provided.')

    f_a_bins = CONFIG.training.fill_factor_bins

    num_volumes = len(volumes)
    validation_data = {k: simple_training_generator(
            v.SubvolumeGenerator(v,
                                 CONFIG.model.block_size,
                                 CONFIG.volume.downsample,
                                 partition=(CONFIG.training.partitions, CONFIG.training.validation_partition)),
            CONFIG.training.batch_size,
            CONFIG.training.validation_size,
            f_a_bins=f_a_bins) for k, v in volumes.iteritems()}
    validation_data = roundrobin(*validation_data.values())

    # Pre-train
    training_data = {k: simple_training_generator(
            v.SubvolumeGenerator(v,
                                 CONFIG.model.block_size,
                                 CONFIG.volume.downsample,
                                 partition=(CONFIG.training.partitions, CONFIG.training.training_partition)),
            CONFIG.training.batch_size,
            CONFIG.training.training_size,
            f_a_bins=f_a_bins) for k, v in volumes.iteritems()}
    training_data = roundrobin(*training_data.values())
    history = ffn.fit_generator(
            training_data,
            samples_per_epoch=CONFIG.training.training_size * num_volumes,
            nb_epoch=CONFIG.training.simple_train_epochs,
            validation_data=validation_data,
            nb_val_samples=CONFIG.training.validation_size * num_volumes)

    # Moving training
    kludges = {k: {'inputs': None, 'outputs': None} for k in volumes.iterkeys()}
    callbacks = [PredictionCopy(kludge) for kludge in kludges.values()]
    callbacks.append(ModelCheckpoint(model_output_filebase + '.hdf5', save_best_only=True))
    if model_checkpoint_file:
        callbacks.append(ModelCheckpoint(model_checkpoint_file))
    callbacks.append(EarlyStopping(patience=CONFIG.training.patience))
    if tensorboard:
        callbacks.append(TensorBoard())

    training_data = {k: moving_training_generator(
            v.SubvolumeGenerator(v,
                                 CONFIG.model.training_fov,
                                 CONFIG.volume.downsample,
                                 partition=(CONFIG.training.partitions, CONFIG.training.training_partition)),
            CONFIG.training.batch_size,
            CONFIG.training.training_size,
            kludges[k],
            f_a_bins=f_a_bins) for k, v in volumes.iteritems()}
    training_data = roundrobin(*training_data.values())
    moving_history = ffn.fit_generator(
            training_data,
            samples_per_epoch=CONFIG.training.training_size * num_volumes,
            nb_epoch=CONFIG.training.total_epochs,
            initial_epoch=CONFIG.training.simple_train_epochs,
            max_q_size=num_volumes,
            nb_worker=1,
            callbacks=callbacks,
            validation_data=validation_data,
            nb_val_samples=CONFIG.training.validation_size * num_volumes)
    extend_keras_history(history, moving_history)

    write_keras_history_to_csv(history, model_output_filebase + '.csv')

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
        fig = plot_history(history)
        fig.savefig(model_output_filebase + '.png')

    return history