# -*- coding: utf-8 -*-
"""Functions for generating training data and training networks."""


from __future__ import division
from __future__ import print_function

import importlib
import itertools
import logging
import random

import matplotlib as mpl
# Use the 'Agg' backend to allow the generation of plots even if no X server
# is available. The matplotlib backend must be set before importing pyplot.
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import six
from six.moves import range as xrange

from keras.callbacks import (
        Callback,
        EarlyStopping,
        ModelCheckpoint,
        TensorBoard,
        )

from .config import CONFIG
from .diluvian import partition_volumes
from .network import compile_network, load_model
from .third_party.multi_gpu import make_parallel
from .util import (
        extend_keras_history,
        get_color_shader,
        pad_dims,
        Roundrobin,
        WrappedViewer,
        write_keras_history_to_csv,
        )
from .volumes import (
        ContrastAugmentGenerator,
        GaussianNoiseAugmentGenerator,
        MirrorAugmentGenerator,
        MissingDataAugmentGenerator,
        PermuteAxesAugmentGenerator,
        )
from .regions import (
        Region,
        mask_to_output_target,
        )


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
    """Keras batch end callback to run prediction on input from a kludge.

    Used to predict masks for FOV moving. Surprisingly this is faster than
    using a custom Keras training function to copy model predictions at the
    same time as gradient updates.
    """
    def __init__(self, kludge, name=None):
        self.kludge = kludge
        self.name = name if name is not None else ''

    def on_batch_end(self, batch, logs={}):
        if self.kludge['inputs'] and self.kludge['outputs'] is None:
            logging.debug('Running prediction kludge {}'.format(self.name))
            self.kludge['outputs'] = self.model.predict(self.kludge['inputs'])


class EarlyAbortException(Exception):
    pass


class EarlyAbort(Callback):
    """Keras epoch end callback that aborts if a metric is above a threshold.

    This is useful when convergence is sensitive to initial conditions and
    models are obviously not useful to continue training after only a few
    epochs. Unlike the early stopping callback, this is considered an
    abnormal termination and throws an exception so that behaviors like
    restarting with a new random seed are possible.
    """
    def __init__(self, monitor='val_loss', threshold_epoch=None, threshold_value=None):
        if threshold_epoch is None or threshold_value is None:
            raise ValueError('Epoch and value to enforce threshold must be provided.')

        self.monitor = monitor
        self.threshold_epoch = threshold_epoch - 1
        self.threshold_value = threshold_value

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.threshold_epoch:
            current = logs.get(self.monitor)
            if current >= self.threshold_value:
                raise EarlyAbortException('Aborted after epoch {} because {} was {} < {}'.format(
                    self.threshold_epoch, self.monitor, current, self.threshold_value))


def augment_subvolume_generator(subvolume_generator):
    """Apply data augmentations to a subvolume generator.

    Parameters
    ----------
    subvolume_generator : diluvian.volumes.SubvolumeGenerator

    Returns
    -------
    diluvian.volumes.SubvolumeGenerator
    """
    gen = subvolume_generator
    for axes in CONFIG.training.augment_permute_axes:
        gen = PermuteAxesAugmentGenerator(gen, axes)
    for axis in CONFIG.training.augment_mirrors:
        gen = MirrorAugmentGenerator(gen, axis)
    for v in CONFIG.training.augment_noise:
        gen = GaussianNoiseAugmentGenerator(gen, v['axis'], v['mul'], v['add'])
    for v in CONFIG.training.augment_missing_data:
        gen = MissingDataAugmentGenerator(gen, v['axis'], v['prob'])
    for v in CONFIG.training.augment_contrast:
        gen = ContrastAugmentGenerator(gen, v['axis'], v['prob'], v['scaling_mean'], v['scaling_std'],
                                       v['center_mean'], v['center_std'])

    return gen


def static_training_generator(subvolumes, batch_size, training_size,
                              f_a_bins=None, reset_generators=True):
    """Generate Keras non-moving training tuples from a subvolume generator.

    Note that this generator is not yet compatible with networks with different
    input and output FOV shapes.

    Parameters
    ----------
    subvolumes : generator of Subvolume
    batch_size : int
    training_size : int
        Total size in samples of a training epoch, after which generators will
        be reset if ``reset_generators`` is true.
    f_a_bins : sequence of float, optional
        Bin boundaries for filling fractions. If provided, sample loss will be
        weighted to increase loss contribution from less-frequent f_a bins.
        Otherwise all samples are weighted equally.
    reset_generators : bool
        Whether to reset subvolume generators at the end of each epoch. If true
        subvolumes will be sampled in the same order each epoch.
    """
    mask_input = np.full(np.append(subvolumes.shape, (1,)), CONFIG.model.v_false, dtype='float32')
    mask_input[tuple(np.array(mask_input.shape) // 2)] = CONFIG.model.v_true
    mask_input = np.tile(mask_input, (batch_size, 1, 1, 1, 1))
    f_a_init = False

    if f_a_bins is not None:
        f_a_init = True
        f_a_counts = np.ones_like(f_a_bins, dtype=np.int64)
    f_as = np.zeros(batch_size)

    sample_num = 0
    while True:
        if sample_num >= training_size:
            f_a_init = False
            if reset_generators:
                subvolumes.reset()
            sample_num = 0

        batch_image_input = [None] * batch_size
        batch_mask_target = [None] * batch_size

        for batch_ind in range(0, batch_size):
            subvolume = subvolumes.next()

            f_as[batch_ind] = subvolume.f_a()
            batch_image_input[batch_ind] = pad_dims(subvolume.image)
            batch_mask_target[batch_ind] = pad_dims(mask_to_output_target(subvolume.label_mask))

        batch_image_input = np.concatenate(batch_image_input)
        batch_mask_target = np.concatenate(batch_mask_target)

        sample_num += batch_size

        if f_a_bins is None:
            yield ({'image_input': batch_image_input,
                    'mask_input': mask_input},
                   [batch_mask_target])
        else:
            f_a_inds = np.digitize(f_as, f_a_bins) - 1
            inds, counts = np.unique(f_a_inds, return_counts=True)
            if f_a_init:
                f_a_counts[inds] += counts.astype(np.int64)
                sample_weights = np.ones(f_as.size, dtype='float64')
            else:
                sample_weights = np.reciprocal(f_a_counts[f_a_inds], dtype='float64') * float(f_as.size)
            yield ({'image_input': batch_image_input,
                    'mask_input': mask_input},
                   [batch_mask_target],
                   sample_weights)


def moving_training_generator(subvolumes, batch_size, training_size, callback_kludge,
                              f_a_bins=None, reset_generators=True):
    """Generate Keras moving FOV training tuples from a subvolume generator.

    Unlike ``static_training_generator``, this generator expects a subvolume
    generator that will provide subvolumes larger than the network FOV, and
    will allow the output of training at one batch to generate moves within
    these subvolumes to produce training data for the subsequent batch.

    Parameters
    ----------
    subvolumes : generator of Subvolume
    batch_size : int
    training_size : int
        Total size in samples of a training epoch, after which generators will
        be reset if ``reset_generators`` is true.
    callback_kludge : dict
        A kludge object to allow this generator to provide inputs and receive
        outputs from the network. See ``diluvian.PredictionCopy``.
    f_a_bins : sequence of float, optional
        Bin boundaries for filling fractions. If provided, sample loss will be
        weighted to increase loss contribution from less-frequent f_a bins.
        Otherwise all samples are weighted equally.
    reset_generators : bool
        Whether to reset subvolume generators at the end of each epoch. If true
        subvolumes will be sampled in the same order each epoch.
    """
    regions = [None] * batch_size
    region_pos = [None] * batch_size
    move_counts = [0] * batch_size
    epoch_move_counts = []
    batch_image_input = [None] * batch_size
    f_a_init = False

    if f_a_bins is not None:
        f_a_init = True
        f_a_counts = np.ones_like(f_a_bins, dtype=np.int64)
    f_as = np.zeros(batch_size)

    sample_num = 0
    while True:
        if sample_num >= training_size:
            f_a_init = False
            if reset_generators:
                subvolumes.reset()
            if len(epoch_move_counts):
                logging.info(' Average moves: %s', sum(epoch_move_counts)/float(len(epoch_move_counts)))
            epoch_move_counts = []
            sample_num = 0

        # Before clearing last batches, reuse them to predict mask outputs
        # for move training. Add mask outputs to regions.
        active_regions = [n for n, region in enumerate(regions) if region is not None]
        if active_regions and callback_kludge['outputs'] is not None:
            for n in active_regions:
                assert np.array_equal(callback_kludge['inputs']['image_input'][n, 0, 0, :, 0],
                                      batch_image_input[n, 0, 0, :, 0])
                regions[n].add_mask(callback_kludge['outputs'][n, :, :, :, 0], region_pos[n])

        batch_image_input = [None] * batch_size
        batch_mask_input = [None] * batch_size
        batch_mask_target = [None] * batch_size

        for r, region in enumerate(regions):
            if region is None or region.queue.empty():
                subvolume = six.next(subvolumes)

                regions[r] = Region.from_subvolume(subvolume)
                region = regions[r]
                epoch_move_counts.append(move_counts[r])
                move_counts[r] = 0
            else:
                move_counts[r] += 1

            block_data = region.get_next_block()

            f_as[r] = subvolume.f_a()
            batch_image_input[r] = pad_dims(block_data['image'])
            batch_mask_input[r] = pad_dims(block_data['mask'])
            batch_mask_target[r] = pad_dims(block_data['target'])
            region_pos[r] = block_data['position']

        batch_image_input = np.concatenate(batch_image_input)
        batch_mask_input = np.concatenate(batch_mask_input)
        batch_mask_target = np.concatenate(batch_mask_target)

        sample_num += batch_size
        inputs = {'image_input': batch_image_input,
                  'mask_input': batch_mask_input}
        callback_kludge['inputs'] = inputs
        callback_kludge['outputs'] = None

        if f_a_bins is None:
            yield (inputs,
                   [batch_mask_target])
        else:
            f_a_inds = np.digitize(f_as, f_a_bins) - 1
            inds, counts = np.unique(f_a_inds, return_counts=True)
            if f_a_init:
                f_a_counts[inds] += counts.astype(np.int64)
                sample_weights = np.ones(f_as.size, dtype='float64')
            else:
                sample_weights = np.reciprocal(f_a_counts[f_a_inds], dtype='float64') * float(f_as.size)
            yield (inputs,
                   [batch_mask_target],
                   sample_weights)


def train_network(
        model_file=None,
        volumes=None,
        static_validation=False,
        model_output_filebase=None,
        model_checkpoint_file=None,
        tensorboard=False,
        viewer=False,
        metric_plot=False):
    random.seed(CONFIG.random_seed)

    if model_file is None:
        factory_mod_name, factory_func_name = CONFIG.network.factory.rsplit('.', 1)
        factory_mod = importlib.import_module(factory_mod_name)
        factory = getattr(factory_mod, factory_func_name)
        ffn = factory(CONFIG.model.input_fov_shape,
                      CONFIG.model.output_fov_shape,
                      CONFIG.network)
    else:
        ffn = load_model(model_file, CONFIG.network)

    # Multi-GPU models are saved as a single-GPU model prior to compilation,
    # so if loading from such a model file it will need to be recompiled.
    if not hasattr(ffn, 'optimizer'):
        if CONFIG.training.num_gpus > 1:
            ffn = make_parallel(ffn, CONFIG.training.num_gpus)
        compile_network(ffn, CONFIG.optimizer)

    if model_output_filebase is None:
        model_output_filebase = 'model_output'

    if volumes is None:
        raise ValueError('Volumes must be provided.')

    CONFIG.to_toml(model_output_filebase + '.toml')

    f_a_bins = CONFIG.training.fill_factor_bins

    training_volumes, validation_volumes = partition_volumes(volumes)

    num_training = len(training_volumes)
    num_validation = len(validation_volumes)

    logging.info('Using {} volumes for training, {} for validation.'.format(num_training, num_validation))

    callbacks = []
    if CONFIG.training.early_abort_epoch is not None and \
       CONFIG.training.early_abort_loss is not None:
        callbacks.append(EarlyAbort(threshold_epoch=CONFIG.training.early_abort_epoch,
                                    threshold_value=CONFIG.training.early_abort_loss))

    validation_kludge = {'inputs': None, 'outputs': None}
    if static_validation:
        validation_shape = CONFIG.model.input_fov_shape
    else:
        validation_shape = CONFIG.model.training_subv_shape
        callbacks.append(PredictionCopy(validation_kludge, 'Validation'))
    validation_gens = [
            augment_subvolume_generator(v.subvolume_generator(shape=validation_shape))
            for v in six.itervalues(validation_volumes)]
    validation_data = moving_training_generator(
            Roundrobin(*validation_gens),
            CONFIG.training.batch_size,
            CONFIG.training.validation_size,
            validation_kludge,
            f_a_bins=f_a_bins,
            reset_generators=True)

    TRAINING_STEPS_PER_EPOCH = CONFIG.training.training_size // CONFIG.training.batch_size
    VALIDATION_STEPS = CONFIG.training.validation_size // CONFIG.training.batch_size

    # Pre-train
    training_gens = [
            augment_subvolume_generator(v.subvolume_generator(shape=CONFIG.model.input_fov_shape))
            for v in six.itervalues(training_volumes)]
    random.shuffle(training_gens)
    # Divide training generators up for workers.
    worker_gens = [
            training_gens[i::CONFIG.training.num_workers]
            for i in xrange(CONFIG.training.num_workers)]
    worker_training_size = CONFIG.training.training_size // CONFIG.training.num_workers
    # Create a training data generator for each worker.
    training_data = [moving_training_generator(
            Roundrobin(*gen),
            CONFIG.training.batch_size,
            worker_training_size,
            {'outputs': None},  # Allows use of moving training gen like static.
            f_a_bins=f_a_bins,
            reset_generators=CONFIG.training.reset_generators) for gen in worker_gens]
    history = ffn.fit_generator(
            Roundrobin(*training_data),
            steps_per_epoch=TRAINING_STEPS_PER_EPOCH,
            epochs=CONFIG.training.static_train_epochs,
            max_q_size=CONFIG.training.num_workers,
            workers=1,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=VALIDATION_STEPS)

    # Moving training
    kludges = [{'inputs': None, 'outputs': None} for _ in range(CONFIG.training.num_workers)]
    callbacks.extend([PredictionCopy(kludge, 'Training {}'.format(n)) for n, kludge in enumerate(kludges)])
    callbacks.append(ModelCheckpoint(model_output_filebase + '.hdf5', save_best_only=True))
    if model_checkpoint_file:
        callbacks.append(ModelCheckpoint(model_checkpoint_file))
    callbacks.append(EarlyStopping(patience=CONFIG.training.patience))
    if tensorboard:
        callbacks.append(TensorBoard())

    training_gens = [
            augment_subvolume_generator(v.subvolume_generator(shape=CONFIG.model.training_subv_shape))
            for v in six.itervalues(training_volumes)]
    random.shuffle(training_gens)
    worker_gens = [
            training_gens[i::CONFIG.training.num_workers]
            for i in xrange(CONFIG.training.num_workers)]
    training_data = [moving_training_generator(
            Roundrobin(*gen),
            CONFIG.training.batch_size,
            worker_training_size,
            kludge,
            f_a_bins=f_a_bins,
            reset_generators=CONFIG.training.reset_generators) for gen, kludge in zip(worker_gens, kludges)]
    moving_history = ffn.fit_generator(
            Roundrobin(*training_data),
            steps_per_epoch=TRAINING_STEPS_PER_EPOCH,
            epochs=CONFIG.training.total_epochs,
            initial_epoch=CONFIG.training.static_train_epochs,
            max_q_size=CONFIG.training.num_workers,
            workers=1,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=VALIDATION_STEPS)
    extend_keras_history(history, moving_history)

    write_keras_history_to_csv(history, model_output_filebase + '.csv')

    if viewer:
        dupe_data = static_training_generator(
                validation_volumes.values()[0].subvolume_generator(shape=CONFIG.model.input_fov_shape),
                CONFIG.training.batch_size,
                CONFIG.training.training_size)
        viz_ex = itertools.islice(dupe_data, 1)

        for inputs, targets in viz_ex:
            viewer = WrappedViewer(voxel_size=list(np.flipud(CONFIG.volume.resolution)))
            viewer.add(inputs['image_input'][0, :, :, :, 0],
                       name='Image')
            viewer.add(inputs['mask_input'][0, :, :, :, 0],
                       name='Mask Input',
                       shader=get_color_shader(2))
            viewer.add(targets[0][0, :, :, :, 0],
                       name='Mask Target',
                       shader=get_color_shader(0))
            output = ffn.predict(inputs)
            viewer.add(output[0, :, :, :, 0],
                       name='Mask Output',
                       shader=get_color_shader(1))

            viewer.print_view_prompt()

    if metric_plot:
        fig = plot_history(history)
        fig.savefig(model_output_filebase + '.png')

    return history
