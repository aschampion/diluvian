# -*- coding: utf-8 -*-


import importlib
import itertools
import logging

import matplotlib as mpl
# Use the 'Agg' backend to allow the generation of plots even if no X server
# is available. The matplotlib backend must be set before importing pyplot.
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import neuroglancer
import numpy as np
import pytoml as toml

from keras.callbacks import (
        Callback,
        EarlyStopping,
        ModelCheckpoint,
        TensorBoard,
        )
from keras.models import load_model

from .config import CONFIG
from .network import compile_network
from .third_party.multi_gpu import make_parallel
from .util import (
        extend_keras_history,
        get_color_shader,
        roundrobin,
        WrappedViewer,
        write_keras_history_to_csv,
        )
from .volumes import (
        HDF5Volume,
        SubvolumeBounds,
        static_training_generator,
        moving_training_generator,
        )
from .regions import Region


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
    def __init__(self, kludge):
        self.kludge = kludge

    def on_batch_end(self, batch, logs={}):
        if self.kludge['inputs'] and self.kludge['outputs'] is None:
            self.kludge['outputs'] = self.model.predict(self.kludge['inputs'])


def generate_subvolume_bounds(filename, volumes, num_bounds, sparse=False):
    if '{volume}' not in filename:
        raise ValueError('CSV filename must contain "{volume}" for volume name replacement.')

    if sparse:
        gen_kwargs = {'sparse_margin': CONFIG.model.training_subv_shape * 4 - 3}
    else:
        gen_kwargs = {'shape': CONFIG.model.training_subv_shape * 4 - 3}
    for k, v in volumes.iteritems():
        bounds = v.downsample(CONFIG.volume.resolution)\
                  .subvolume_bounds_generator(**gen_kwargs)
        bounds = itertools.islice(bounds, num_bounds)
        SubvolumeBounds.iterable_to_csv(bounds, filename.format(volume=k))


def fill_subvolume_with_model(
        model_file,
        subvolume,
        background_label_id=0,
        bias=True,
        move_batch_size=1,
        max_bodies=None):
    # Create an output label volume.
    prediction = np.full_like(subvolume.image, background_label_id, dtype=np.uint64)
    # Create a conflict count volume that tracks locations where segmented
    # bodies overlap. For now the first body takes precedence in the
    # predicted labels.
    conflict_count = np.full_like(prediction, 0, dtype=np.uint32)

    # Generate seeds from volume.
    # For now just use a uniform grid.
    seeds = []
    grid_size = (CONFIG.model.output_fov_shape - 1) / 2
    for x in range(grid_size[0], prediction.shape[0], grid_size[0]):
        for y in range(grid_size[1], prediction.shape[1], grid_size[1]):
            for z in range(grid_size[2], prediction.shape[2], grid_size[2]):
                seeds.append(np.array([x, y, z], dtype=np.int32))

    model = load_model(model_file)

    label_id = 0
    # For each seed, create region, fill, threshold, and merge to output volume.
    for seed_idx, seed in enumerate(seeds):
        logging.debug('Processing seed at %s', np.array_str(seed))
        if prediction[seed[0], seed[1], seed[2]] != background_label_id:
            # This seed has already been filled.
            continue

        # Flood-fill and get resulting mask.
        # Allow reading outside the image volume bounds to allow segmentation
        # to fill all the way to the boundary.
        region = Region(subvolume.image, seed_vox=seed, block_padding='reflect')
        region.bias_against_merge = bias
        region.fill(model,
                    move_batch_size=move_batch_size,
                    verbose=True)
        body = region.to_body()

        # Generate a label ID for this region.
        label_id += 1
        if label_id == background_label_id:
            label_id += 1

        logging.debug('Adding body to prediction label volume.')
        conflict_count[np.logical_and(prediction != background_label_id, body.mask)] += 1
        prediction[np.logical_and(prediction == background_label_id, body.mask)] = label_id
        logging.info('Filled seed %s/%s (%s) with %s voxels labeled %s.',
                     seed_idx, len(seeds), np.array_str(seed), np.count_nonzero(body.mask), label_id)

        if max_bodies and label_id >= max_bodies:
            break

    return prediction, conflict_count


def fill_volumes_with_model(
        model_file,
        volumes,
        filename,
        viewer=False,
        **kwargs):
    if '{volume}' not in filename:
        raise ValueError('HDF5 filename must contain "{volume}" for volume name replacement.')

    for volume_name, volume in volumes.iteritems():
        logging.info('Filling volume %s...', volume_name)
        volume = volume.downsample(CONFIG.volume.resolution)
        volume = volume.get_subvolume(SubvolumeBounds(start=np.zeros(3, dtype=np.int64), stop=volume.shape))
        prediction, conflict_count = fill_subvolume_with_model(model_file, volume, **kwargs)

        volume_filename = filename.format(volume=volume_name)
        config = HDF5Volume.write_file(
                volume_filename + '.hdf5',
                CONFIG.volume.resolution,
                label_data=prediction)
        config['name'] = volume_name + ' segmentation'
        with open(volume_filename + '.toml', 'wb') as tomlfile:
            tomlfile.write(str(toml.dumps({'dataset': [config]})))

        if viewer:
            viewer = WrappedViewer(voxel_size=list(np.flipud(CONFIG.volume.resolution)))
            viewer.add(volume.image, name='Image')
            viewer.add(prediction, name='Labels')
            viewer.add(conflict_count, name='Conflicts')

            viewer.print_view_prompt()


def fill_region_with_model(
        model_file,
        volumes=None,
        bounds_input_file=None,
        bias=True,
        move_batch_size=1,
        max_moves=None,
        multi_gpu_model_kludge=None,
        sparse=False):
    if volumes is None:
        raise ValueError('Volumes must be provided.')

    if bounds_input_file is not None:
        gen_kwargs = {
                k: {'bounds_generator': iter(SubvolumeBounds.iterable_from_csv(bounds_input_file.format(volume=k)))}
                for k in volumes.iterkeys()}
    else:
        if sparse:
            gen_kwargs = {
                    k: {'sparse_margin': CONFIG.model.training_subv_shape * 4 - 3}
                    for k in volumes.iterkeys()}
        else:
            gen_kwargs = {
                    k: {'shape': CONFIG.model.training_subv_shape * 4 - 3}
                    for k in volumes.iterkeys()}
    regions = roundrobin(*[
            Region.from_subvolume_generator(
                v.downsample(CONFIG.volume.resolution)
                 .subvolume_generator(**gen_kwargs[k]))
            for k, v in volumes.iteritems()])

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
        while True:
            s = raw_input("Press Enter to continue, v to open in browser, a to export animation, q to quit...")
            if s == 'q':
                return
            elif s == 'a':
                region_copy = region.unfilled_copy()
                # Must assign the animation to a variable so that it is not GCed.
                ani = region_copy.fill_animation(model, 'export.mp4', verbose=True) # noqa
                s = raw_input("Press Enter when animation is complete...")
            elif s == 's':
                body = region.to_body()
                body.to_swc('{}.swc'.format('_'.join(map(str, tuple(body.seed)))))
            elif s == 'v':
                viewer.open_in_browser()
            else:
                break


def train_network(
        model_file=None,
        volumes=None,
        static_validation=True,
        reset_generators_each_epoch=True,
        model_output_filebase=None,
        model_checkpoint_file=None,
        tensorboard=False,
        viewer=False,
        metric_plot=False):
    if model_file is None:
        factory_mod_name, factory_func_name = CONFIG.network.factory.rsplit('.', 1)
        factory_mod = importlib.import_module(factory_mod_name)
        factory = getattr(factory_mod, factory_func_name)
        ffn = factory(CONFIG.model.input_fov_shape,
                      CONFIG.model.output_fov_shape,
                      CONFIG.network)
    else:
        ffn = load_model(model_file)

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

    num_volumes = len(volumes)

    training_volumes = {
            k: v.partition(CONFIG.training.partitions, CONFIG.training.training_partition)
                .downsample(CONFIG.volume.resolution)
            for k, v in volumes.iteritems()}
    validation_volumes = {
            k: v.partition(CONFIG.training.partitions, CONFIG.training.validation_partition)
                .downsample(CONFIG.volume.resolution)
            for k, v in volumes.iteritems()}

    if static_validation:
        validation_data = {k: moving_training_generator(
                v.subvolume_generator(shape=CONFIG.model.input_fov_shape),
                CONFIG.training.batch_size,
                CONFIG.training.validation_size,
                {'outputs': None},  # Allows use of moving training gen like static.
                f_a_bins=f_a_bins,
                reset_generators=True) for k, v in validation_volumes.iteritems()}
    else:
        validation_kludges = {k: {'inputs': None, 'outputs': None} for k in volumes.iterkeys()}
        validation_data = {k: moving_training_generator(
                v.subvolume_generator(shape=CONFIG.model.training_subv_shape),
                CONFIG.training.batch_size,
                CONFIG.training.validation_size,
                validation_kludges[k],
                f_a_bins=f_a_bins,
                reset_generators=True) for k, v in validation_volumes.iteritems()}
    validation_data = roundrobin(*validation_data.values())

    # Pre-train
    training_data = {k: moving_training_generator(
            v.subvolume_generator(shape=CONFIG.model.input_fov_shape),
            CONFIG.training.batch_size,
            CONFIG.training.training_size,
            {'outputs': None},  # Allows use of moving training gen like static.
            f_a_bins=f_a_bins,
            reset_generators=reset_generators_each_epoch) for k, v in training_volumes.iteritems()}
    training_data = roundrobin(*training_data.values())
    history = ffn.fit_generator(
            training_data,
            samples_per_epoch=CONFIG.training.training_size * num_volumes,
            nb_epoch=CONFIG.training.static_train_epochs,
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
            v.subvolume_generator(shape=CONFIG.model.training_subv_shape),
            CONFIG.training.batch_size,
            CONFIG.training.training_size,
            kludges[k],
            f_a_bins=f_a_bins,
            reset_generators=reset_generators_each_epoch) for k, v in training_volumes.iteritems()}
    training_data = roundrobin(*training_data.values())
    moving_history = ffn.fit_generator(
            training_data,
            samples_per_epoch=CONFIG.training.training_size * num_volumes,
            nb_epoch=CONFIG.training.total_epochs,
            initial_epoch=CONFIG.training.static_train_epochs,
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
        dupe_data = static_training_generator(
                volumes[list(volumes.keys())[0]].subvolume_generator(shape=CONFIG.model.input_fov_shape),
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


def view_volumes(volumes):
    viewer = WrappedViewer()

    for volume_name, volume in volumes.iteritems():
        viewer.add(volume.image_data,
                   name='{} (Image)'.format(volume_name),
                   voxel_size=list(np.flipud(volume.resolution)))
        viewer.add(volume.label_data,
                   name='{} (Labels)'.format(volume_name),
                   voxel_size=list(np.flipud(volume.resolution)))

    viewer.print_view_prompt()
