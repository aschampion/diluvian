# -*- coding: utf-8 -*-
"""Functions for generating training data and training networks."""


from __future__ import division
from __future__ import print_function

import collections
import importlib
import itertools
import logging
import random
import types

import matplotlib as mpl
# Use the 'Agg' backend to allow the generation of plots even if no X server
# is available. The matplotlib backend must be set before importing pyplot.
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import six
from six.moves import range as xrange

import keras.backend as K
from keras.callbacks import (
        Callback,
        EarlyStopping,
        ModelCheckpoint,
        TensorBoard,
        )
from keras.engine import Model

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
        ErodedMaskGenerator,
        GaussianNoiseAugmentGenerator,
        MaskedArtifactAugmentGenerator,
        MirrorAugmentGenerator,
        MissingDataAugmentGenerator,
        PermuteAxesAugmentGenerator,
        RelabelSeedComponentGenerator,
        )
from .regions import (
        Region,
        )


def plot_history(history):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    fig.suptitle('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'], loc='upper right')

    return fig


def patch_prediction_copy(model):
    """Patch a Keras model to copy outputs to a kludge during training.

    This is necessary for mask updates to a region during training.

    Parameters
    ----------
    model : keras.engine.Model
    """
    model._orig_train_on_batch = model.train_on_batch

    def train_on_batch(self, x, y, **kwargs):
        kludge = x.pop('kludge', None)
        outputs = self._orig_train_on_batch(x, y, **kwargs)
        kludge['outputs'] = outputs.pop()
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    model.train_on_batch = types.MethodType(train_on_batch, model, Model)

    model._orig_test_on_batch = model.test_on_batch

    def test_on_batch(self, x, y, **kwargs):
        kludge = x.pop('kludge', None)
        outputs = self._orig_test_on_batch(x, y, **kwargs)
        kludge['outputs'] = outputs.pop()
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    model.test_on_batch = types.MethodType(test_on_batch, model, Model)

    # Below is copied and modified from Keras Model._make_train_function.
    # The only change is the addition of `self.outputs` to the train function.
    def _make_train_function(self):
        if not hasattr(self, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.train_function is None:
            inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]

            with K.name_scope('training'):
                with K.name_scope(self.optimizer.__class__.__name__):
                    training_updates = self.optimizer.get_updates(
                        params=self._collected_trainable_weights,
                        loss=self.total_loss)
                updates = self.updates + training_updates
                # Gets loss and metrics. Updates weights at each call.
                self.train_function = K.function(inputs,
                                                 [self.total_loss] + self.metrics_tensors + self.outputs,
                                                 updates=updates,
                                                 name='train_function',
                                                 **self._function_kwargs)

    model._make_train_function = types.MethodType(_make_train_function, model, Model)

    def _make_test_function(self):
        if not hasattr(self, 'test_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.test_function is None:
            inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]
            # Return loss and metrics, no gradient updates.
            # Does update the network states.
            self.test_function = K.function(inputs,
                                            [self.total_loss] + self.metrics_tensors + self.outputs,
                                            updates=self.state_updates,
                                            name='test_function',
                                            **self._function_kwargs)

    model._make_test_function = types.MethodType(_make_test_function, model, Model)


class GeneratorReset(Callback):
    """Keras epoch end callback to reset prediction copy kludges.
    """
    def __init__(self, gens):
        self.gens = gens

    def on_epoch_end(self, epoch, logs=None):
        for gen in self.gens:
            gen.reset()


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
                raise EarlyAbortException('Aborted after epoch {} because {} was {} >= {}'.format(
                    self.threshold_epoch, self.monitor, current, self.threshold_value))


def preprocess_subvolume_generator(subvolume_generator):
    """Apply non-augmentation preprocessing to a subvolume generator.

    Parameters
    ----------
    subvolume_generator : diluvian.volumes.SubvolumeGenerator

    Returns
    -------
    diluvian.volumes.SubvolumeGenerator
    """
    gen = subvolume_generator
    if np.any(CONFIG.training.label_erosion):
        gen = ErodedMaskGenerator(gen, CONFIG.training.label_erosion)
    if CONFIG.training.relabel_seed_component:
        gen = RelabelSeedComponentGenerator(gen)

    return gen


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
        gen = PermuteAxesAugmentGenerator(gen, CONFIG.training.augment_use_both, axes)
    for axis in CONFIG.training.augment_mirrors:
        gen = MirrorAugmentGenerator(gen, CONFIG.training.augment_use_both, axis)
    for v in CONFIG.training.augment_noise:
        gen = GaussianNoiseAugmentGenerator(gen, CONFIG.training.augment_use_both, v['axis'], v['mul'], v['add'])
    for v in CONFIG.training.augment_artifacts:
        if 'cache' not in v:
            v['cache'] = {}
        gen = MaskedArtifactAugmentGenerator(gen, CONFIG.training.augment_use_both,
                                             v['axis'], v['prob'], v['volume_file'], v['cache'])
    for v in CONFIG.training.augment_missing_data:
        gen = MissingDataAugmentGenerator(gen, CONFIG.training.augment_use_both, v['axis'], v['prob'])
    for v in CONFIG.training.augment_contrast:
        gen = ContrastAugmentGenerator(gen, CONFIG.training.augment_use_both, v['axis'], v['prob'],
                                       v['scaling_mean'], v['scaling_std'],
                                       v['center_mean'], v['center_std'])

    return gen


class MovingTrainingGenerator(six.Iterator):
    """Generate Keras moving FOV training tuples from a subvolume generator.

    This generator expects a subvolume generator that will provide subvolumes
    larger than the network FOV, and will allow the output of training at one
    batch to generate moves within these subvolumes to produce training data
    for the subsequent batch.

    Parameters
    ----------
    subvolumes : generator of Subvolume
    batch_size : int
    kludge : dict
        A kludge object to allow this generator to provide inputs and receive
        outputs from the network.
        See ``diluvian.training.patch_prediction_copy``.
    f_a_bins : sequence of float, optional
        Bin boundaries for filling fractions. If provided, sample loss will be
        weighted to increase loss contribution from less-frequent f_a bins.
        Otherwise all samples are weighted equally.
    reset_generators : bool
        Whether to reset subvolume generators when this generator is reset.
        If true subvolumes will be sampled in the same order each epoch.
    """
    def __init__(self, subvolumes, batch_size, kludge,
                 f_a_bins=None, reset_generators=True):
        self.subvolumes = subvolumes
        self.batch_size = batch_size
        self.kludge = kludge
        self.reset_generators = reset_generators

        self.regions = [None] * batch_size
        self.region_pos = [None] * batch_size
        self.move_counts = [0] * batch_size
        self.epoch_move_counts = []
        self.batch_image_input = [None] * batch_size

        self.f_a_bins = f_a_bins
        self.f_a_init = False
        if f_a_bins is not None:
            self.f_a_init = True
            self.f_a_counts = np.ones_like(f_a_bins, dtype=np.int64)
        self.f_as = np.zeros(batch_size)

    def __iter__(self):
        return self

    def reset(self):
        self.f_a_init = False
        if self.reset_generators:
            self.subvolumes.reset()
            self.regions = [None] * self.batch_size
            self.kludge['inputs'] = None
            self.kludge['outputs'] = None
        if len(self.epoch_move_counts):
            logging.info(' Average moves (%s): %s',
                         self.subvolumes.name,
                         sum(self.epoch_move_counts)/float(len(self.epoch_move_counts)))
        self.epoch_move_counts = []

    def __next__(self):
        # Before clearing last batches, reuse them to predict mask outputs
        # for move training. Add mask outputs to regions.
        active_regions = [n for n, region in enumerate(self.regions) if region is not None]
        if active_regions and self.kludge['outputs'] is not None and self.kludge['inputs'] is not None:
            for n in active_regions:
                assert np.array_equal(self.kludge['inputs'][n, :],
                                      self.batch_image_input[n, 0, 0, :, 0])
                self.regions[n].add_mask(self.kludge['outputs'][n, :, :, :, 0], self.region_pos[n])

        self.batch_image_input = [None] * self.batch_size
        batch_mask_input = [None] * self.batch_size
        batch_mask_target = [None] * self.batch_size

        for r, region in enumerate(self.regions):
            block_data = region.get_next_block() if region is not None else None
            if block_data is None:
                while block_data is None:
                    subvolume = six.next(self.subvolumes)
                    self.f_as[r] = subvolume.f_a()

                    self.regions[r] = Region.from_subvolume(subvolume)
                    region = self.regions[r]
                    self.epoch_move_counts.append(self.move_counts[r])
                    self.move_counts[r] = 0
                    block_data = region.get_next_block()
            else:
                self.move_counts[r] += 1

            self.batch_image_input[r] = pad_dims(block_data['image'])
            batch_mask_input[r] = pad_dims(block_data['mask'])
            batch_mask_target[r] = pad_dims(block_data['target'])
            self.region_pos[r] = block_data['position']

        self.batch_image_input = np.concatenate(self.batch_image_input)
        batch_mask_input = np.concatenate(batch_mask_input)
        batch_mask_target = np.concatenate(batch_mask_target)

        inputs = collections.OrderedDict({'image_input': self.batch_image_input,
                                          'mask_input': batch_mask_input})
        inputs = collections.OrderedDict({'image_input': self.batch_image_input,
                                          'mask_input': batch_mask_input})
        inputs['kludge'] = self.kludge
        # These inputs are only necessary for assurance the correct FOV is updated.
        self.kludge['inputs'] = self.batch_image_input[:, 0, 0, :, 0].copy()
        self.kludge['outputs'] = None

        if self.f_a_bins is None:
            return (inputs,
                    [batch_mask_target])
        else:
            f_a_inds = np.digitize(self.f_as, self.f_a_bins) - 1
            inds, counts = np.unique(f_a_inds, return_counts=True)
            if self.f_a_init:
                self.f_a_counts[inds] += counts.astype(np.int64)
                sample_weights = np.ones(self.f_as.size, dtype=np.float64)
            else:
                sample_weights = np.reciprocal(self.f_a_counts[f_a_inds], dtype=np.float64) * float(self.f_as.size)
            return (inputs,
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

    patch_prediction_copy(ffn)

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

    validation_kludges = [{'inputs': None, 'outputs': None} for _ in range(CONFIG.training.num_workers)]
    output_margin = np.floor_divide(CONFIG.model.input_fov_shape - CONFIG.model.output_fov_shape, 2)
    if static_validation:
        validation_shape = CONFIG.model.input_fov_shape
    else:
        validation_shape = CONFIG.model.training_subv_shape
    validation_gens = [
            preprocess_subvolume_generator(
                    v.subvolume_generator(shape=validation_shape,
                                          label_margin=output_margin))
            for v in six.itervalues(validation_volumes)]
    if CONFIG.training.augment_validation:
        validation_gens = map(augment_subvolume_generator, validation_gens)
    # Divide training generators up for workers.
    validation_worker_gens = [
            validation_gens[i::CONFIG.training.num_workers]
            for i in xrange(CONFIG.training.num_workers)]
    validation_worker_gens = [g for g in validation_worker_gens if len(g) > 0]
    logging.debug('# of validation workers: %s', len(validation_worker_gens))
    validation_data = [MovingTrainingGenerator(
            Roundrobin(*gen, name='validation inner {}'.format(i)),
            CONFIG.training.batch_size,
            # worker_validation_size,
            kludge,
            f_a_bins=f_a_bins,
            reset_generators=True)
            for i, (gen, kludge) in enumerate(zip(validation_worker_gens, validation_kludges))]
    callbacks.append(GeneratorReset(validation_data))

    TRAINING_STEPS_PER_EPOCH = CONFIG.training.training_size // CONFIG.training.batch_size
    VALIDATION_STEPS = CONFIG.training.validation_size // CONFIG.training.batch_size

    # Pre-train
    training_gens = [
            augment_subvolume_generator(
                    preprocess_subvolume_generator(
                            v.subvolume_generator(shape=CONFIG.model.input_fov_shape,
                                                  label_margin=output_margin)))
            for v in six.itervalues(training_volumes)]
    random.shuffle(training_gens)
    # Divide training generators up for workers.
    worker_gens = [
            training_gens[i::CONFIG.training.num_workers]
            for i in xrange(CONFIG.training.num_workers)]
    # Some workers may not receive any generators.
    worker_gens = [g for g in worker_gens if len(g) > 0]
    logging.debug('# of training workers: %s', len(worker_gens))
    # Create a training data generator for each worker.
    training_data = [MovingTrainingGenerator(
            Roundrobin(*gen, name='static training inner {}'.format(i)),
            CONFIG.training.batch_size,
            {'outputs': None},  # Allows use of moving training gen like static.
            f_a_bins=f_a_bins,
            reset_generators=CONFIG.training.reset_generators)
            for i, gen in enumerate(worker_gens)]
    training_reset_callback = GeneratorReset(training_data)
    callbacks.append(training_reset_callback)
    history = ffn.fit_generator(
            Roundrobin(*training_data, name='static training outer'),
            steps_per_epoch=TRAINING_STEPS_PER_EPOCH,
            epochs=CONFIG.training.static_train_epochs,
            max_queue_size=len(worker_gens) - 1,
            workers=1,
            callbacks=callbacks,
            validation_data=Roundrobin(*validation_data, name='validation out'),
            validation_steps=VALIDATION_STEPS)

    # Moving training
    kludges = [{'inputs': None, 'outputs': None} for _ in range(CONFIG.training.num_workers)]
    callbacks.append(ModelCheckpoint(model_output_filebase + '.hdf5', save_best_only=True))
    if model_checkpoint_file:
        callbacks.append(ModelCheckpoint(model_checkpoint_file))
    callbacks.append(EarlyStopping(patience=CONFIG.training.patience))
    # Activation histograms and weight images for TensorBoard will not work
    # because the Keras callback does not currently support validation data
    # generators.
    if tensorboard:
        callbacks.append(TensorBoard())

    training_gens = [
            augment_subvolume_generator(
                    preprocess_subvolume_generator(
                            v.subvolume_generator(shape=CONFIG.model.training_subv_shape,
                                                  label_margin=output_margin)))
            for v in six.itervalues(training_volumes)]
    random.shuffle(training_gens)
    worker_gens = [
            training_gens[i::CONFIG.training.num_workers]
            for i in xrange(CONFIG.training.num_workers)]
    worker_gens = [g for g in worker_gens if len(g) > 0]
    training_data = [MovingTrainingGenerator(
            Roundrobin(*gen, name='moving training inner {}'.format(i)),
            CONFIG.training.batch_size,
            kludge,
            f_a_bins=f_a_bins,
            reset_generators=CONFIG.training.reset_generators)
            for i, (gen, kludge) in enumerate(zip(worker_gens, kludges))]
    callbacks.remove(training_reset_callback)
    training_reset_callback = GeneratorReset(training_data)
    callbacks.append(training_reset_callback)
    moving_history = ffn.fit_generator(
            Roundrobin(*training_data, name='moving training outer'),
            steps_per_epoch=TRAINING_STEPS_PER_EPOCH,
            epochs=CONFIG.training.total_epochs,
            initial_epoch=CONFIG.training.static_train_epochs,
            max_queue_size=len(worker_gens) - 1,
            workers=1,
            callbacks=callbacks,
            validation_data=Roundrobin(*validation_data, name='validation out'),
            validation_steps=VALIDATION_STEPS)
    extend_keras_history(history, moving_history)

    write_keras_history_to_csv(history, model_output_filebase + '.csv')

    if viewer:
        viz_ex = itertools.islice(validation_data, 1)

        for inputs, targets in viz_ex:
            viewer = WrappedViewer(voxel_size=list(np.flipud(CONFIG.volume.resolution)))
            output_offset = np.array(inputs['image_input'].shape[1:4]) - np.array(targets[0].shape[1:4])
            output_offset = np.flipud(output_offset // 2)
            viewer.add(inputs['image_input'][0, :, :, :, 0],
                       name='Image')
            viewer.add(inputs['mask_input'][0, :, :, :, 0],
                       name='Mask Input',
                       shader=get_color_shader(2))
            viewer.add(targets[0][0, :, :, :, 0],
                       name='Mask Target',
                       shader=get_color_shader(0),
                       voxel_offset=output_offset)
            output = ffn.predict_on_batch(inputs)
            viewer.add(output[0, :, :, :, 0],
                       name='Mask Output',
                       shader=get_color_shader(1),
                       voxel_offset=output_offset)

            viewer.print_view_prompt()

    if metric_plot:
        fig = plot_history(history)
        fig.savefig(model_output_filebase + '.png')

    return history
