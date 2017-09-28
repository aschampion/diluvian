# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

from collections import deque
import itertools
import logging
from multiprocessing import (
        Manager,
        Process,
        )
import os
import random
import re

import numpy as np
import pytoml as toml
import six
from six.moves import input as raw_input
from tqdm import tqdm

from .config import CONFIG
from . import preprocessing
from .util import (
        get_color_shader,
        Roundrobin,
        WrappedViewer,
        )
from .volumes import (
        HDF5Volume,
        SubvolumeBounds,
        )
from .regions import Region


def generate_subvolume_bounds(filename, volumes, num_bounds, sparse=False, moves=None):
    if '{volume}' not in filename:
        raise ValueError('CSV filename must contain "{volume}" for volume name replacement.')

    if moves is None:
        moves = 5
    else:
        moves = np.asarray(moves)
    subv_shape = CONFIG.model.input_fov_shape + CONFIG.model.move_step * 2 * moves

    if sparse:
        gen_kwargs = {'sparse_margin': subv_shape}
    else:
        gen_kwargs = {'shape': subv_shape}
    for k, v in six.iteritems(volumes):
        bounds = v.downsample(CONFIG.volume.resolution)\
                  .subvolume_bounds_generator(**gen_kwargs)
        bounds = itertools.islice(bounds, num_bounds)
        SubvolumeBounds.iterable_to_csv(bounds, filename.format(volume=k))


def fill_volume_with_model(
        model_file,
        volume,
        resume_prediction=None,
        seed_generator='sobel',
        background_label_id=0,
        bias=True,
        move_batch_size=1,
        max_moves=None,
        max_bodies=None,
        num_workers=CONFIG.training.num_gpus,
        worker_prequeue=1,
        filter_seeds_by_mask=True,
        reject_non_seed_components=True,
        reject_early_termination=False,
        remask_interval=None,
        shuffle_seeds=True):
    subvolume = volume.get_subvolume(SubvolumeBounds(start=np.zeros(3, dtype=np.int64), stop=volume.shape))
    # Create an output label volume.
    if resume_prediction is None:
        prediction = np.full_like(subvolume.image, background_label_id, dtype=np.uint64)
        label_id = 0
    else:
        if resume_prediction.shape != subvolume.image.shape:
            raise ValueError('Resume volume prediction is wrong shape.')
        prediction = resume_prediction
        prediction.flags.writeable = True
        label_id = prediction.max()
    # Create a conflict count volume that tracks locations where segmented
    # bodies overlap. For now the first body takes precedence in the
    # predicted labels.
    conflict_count = np.full_like(prediction, 0, dtype=np.uint32)

    def worker(worker_id, set_devices, model_file, image, seeds, results, lock, revoked):
        lock.acquire()
        import tensorflow as tf

        if set_devices:
            # Only make one GPU visible to Tensorflow so that it does not allocate
            # all available memory on all devices.
            # See: https://stackoverflow.com/questions/37893755
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_id)

        with tf.device('/gpu:0'):
            # Late import to avoid Keras import until TF bindings are set.
            from .network import load_model

            logging.debug('Worker %s: loading model', worker_id)
            model = load_model(model_file, CONFIG.network)
        lock.release()

        while True:
            seed = seeds.get(True)

            if not isinstance(seed, np.ndarray):
                logging.debug('Worker %s: got DONE', worker_id)
                break

            def stopping_callback(region):
                stop = False
                lock.acquire()
                if tuple(seed) in revoked:
                    revoked.remove(tuple(seed))
                    stop = True
                lock.release()
                if reject_non_seed_components and \
                   region.bias_against_merge and \
                   region.mask[tuple(region.seed_vox)] < 0.5:
                    stop = True
                return stop

            logging.debug('Worker %s: got seed %s', worker_id, np.array_str(seed))

            # Flood-fill and get resulting mask.
            # Allow reading outside the image volume bounds to allow segmentation
            # to fill all the way to the boundary.
            region = Region(image, seed_vox=seed, sparse_mask=True, block_padding='reflect')
            region.bias_against_merge = bias
            early_termination = region.fill(
                    model,
                    move_batch_size=move_batch_size,
                    max_moves=max_moves,
                    progress=1 + worker_id,
                    stopping_callback=stopping_callback,
                    remask_interval=remask_interval)
            if reject_early_termination and early_termination:
                body = None
            else:
                body = region.to_body()
            logging.debug('Worker %s: seed %s filled', worker_id, np.array_str(seed))

            results.put((seed, body))

    # Generate seeds from volume.
    generator = preprocessing.SEED_GENERATORS[seed_generator]
    seeds = generator(subvolume.image, CONFIG.volume.resolution)

    if filter_seeds_by_mask and volume.mask_data is not None:
        seeds = [s for s in seeds if volume.mask_data[tuple(volume.world_coord_to_local(s))]]

    pbar = tqdm(desc='Seed queue', total=len(seeds), miniters=1, smoothing=0.0)
    num_seeds = len(seeds)
    if shuffle_seeds:
        random.shuffle(seeds)
    seeds = iter(seeds)

    manager = Manager()
    # Queue of seeds to be picked up by workers.
    seed_queue = manager.Queue()
    # Queue of results from workers.
    results_queue = manager.Queue()
    # Dequeue of seeds that were put in seed_queue but have not yet been
    # combined by the main process.
    dispatched_seeds = deque()
    # Seeds that were placed in seed_queue but subsequently covered by other
    # results before their results have been processed. This allows workers to
    # abort working on these seeds by checking this list.
    revoked_seeds = manager.list()
    # Results that have been received by the main process but have not yet
    # been combined because they were not received in the dispatch order.
    unordered_results = {}

    def queue_next_seed():
        total = 0
        for seed in seeds:
            if prediction[seed[0], seed[1], seed[2]] != background_label_id:
                # This seed has already been filled.
                total += 1
                continue
            dispatched_seeds.append(seed)
            seed_queue.put(seed)

            break

        return total

    processed_seeds = 1
    for _ in range(min(num_seeds, num_workers * worker_prequeue)):
        processed_seeds += queue_next_seed()

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        set_devices = False
        num_workers = 1
        logging.warn('Environment variable CUDA_VISIBLE_DEVICES is set, so only one worker can be used.\n'
                     'See https://github.com/aschampion/diluvian/issues/11')
    else:
        set_devices = True

    workers = []
    loading_lock = manager.Lock()
    for worker_id in range(num_workers):
        w = Process(target=worker, args=(worker_id, set_devices, model_file, subvolume.image,
                                         seed_queue, results_queue, loading_lock, revoked_seeds))
        w.start()
        workers.append(w)

    # For each seed, create region, fill, threshold, and merge to output volume.
    while dispatched_seeds:
        expected_seed = dispatched_seeds.popleft()
        logging.debug('Expecting seed %s', np.array_str(expected_seed))

        if tuple(expected_seed) in unordered_results:
            logging.debug('Expected seed %s is in old results', np.array_str(expected_seed))
            seed = expected_seed
            body = unordered_results[tuple(seed)]
            del unordered_results[tuple(seed)]

        else:
            seed, body = results_queue.get(True)
            processed_seeds += queue_next_seed()

            while not np.array_equal(seed, expected_seed):
                logging.debug('Seed %s is early, stashing', np.array_str(seed))
                unordered_results[tuple(seed)] = body
                seed, body = results_queue.get(True)
                processed_seeds += queue_next_seed()

        logging.debug('Processing seed at %s', np.array_str(seed))
        pbar.set_description('Seed ' + np.array_str(seed))
        pbar.update(processed_seeds)

        if prediction[seed[0], seed[1], seed[2]] != background_label_id:
            # This seed has already been filled.
            logging.debug('Seed (%s) was filled but has been covered in the meantime.',
                          np.array_str(seed))
            loading_lock.acquire()
            if tuple(seed) in revoked_seeds:
                revoked_seeds.remove(tuple(seed))
            loading_lock.release()
            continue

        if body is None:
            logging.debug('Body was None.')
            continue

        if reject_non_seed_components and not body.is_seed_in_mask():
            logging.debug('Seed (%s) is not in its body.', np.array_str(seed))
            continue

        if reject_non_seed_components:
            mask, bounds = body.get_seeded_component(CONFIG.postprocessing.closing_shape)
        else:
            mask, bounds = body._get_bounded_mask()

        body_size = np.count_nonzero(mask)

        if body_size == 0:
            logging.debug('Body was empty.')
            continue

        # Generate a label ID for this region.
        label_id += 1
        if label_id == background_label_id:
            label_id += 1

        logging.debug('Adding body to prediction label volume.')
        bounds_shape = map(slice, bounds[0], bounds[1])
        prediction_mask = prediction[bounds_shape] == background_label_id
        for seed in dispatched_seeds:
            if np.all(bounds[0] <= seed) and np.all(bounds[1] > seed) and mask[tuple(seed - bounds[0])]:
                loading_lock.acquire()
                if tuple(seed) not in revoked_seeds:
                    revoked_seeds.append(tuple(seed))
                loading_lock.release()
        conflict_count[bounds_shape][np.logical_and(np.logical_not(prediction_mask), mask)] += 1
        prediction[bounds_shape][np.logical_and(prediction_mask, mask)] = label_id
        logging.info('Filled seed (%s) with %s voxels labeled %s.',
                     np.array_str(seed), body_size, label_id)

        if max_bodies and label_id >= max_bodies:
            # Drain the queues.
            while not seed_queue.empty():
                seed_queue.get_nowait()
            break

    for _ in range(num_workers):
        seed_queue.put('DONE')
    for wid, worker in enumerate(workers):
        worker.join()
    manager.shutdown()

    return prediction, conflict_count


def fill_volumes_with_model(
        model_file,
        volumes,
        filename,
        resume_filename=None,
        viewer=False,
        **kwargs):
    if '{volume}' not in filename:
        raise ValueError('HDF5 filename must contain "{volume}" for volume name replacement.')
    if resume_filename is not None and '{volume}' not in resume_filename:
        raise ValueError('TOML resume filename must contain "{volume}" for volume name replacement.')

    for volume_name, volume in six.iteritems(volumes):
        logging.info('Filling volume %s...', volume_name)
        volume = volume.downsample(CONFIG.volume.resolution)
        if resume_filename is not None:
            resume_volume_filename = resume_filename.format(volume=volume_name)
            resume_volume = six.next(six.itervalues(HDF5Volume.from_toml(resume_volume_filename)))
            resume_prediction = resume_volume.to_memory_volume().label_data
        else:
            resume_prediction = None
        prediction, conflict_count = fill_volume_with_model(
                model_file,
                volume,
                resume_prediction=resume_prediction,
                **kwargs)

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
            subvolume = volume.get_subvolume(SubvolumeBounds(start=np.zeros(3, dtype=np.int64), stop=volume.shape))
            viewer.add(subvolume.image, name='Image')
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
        remask_interval=None,
        sparse=False,
        moves=None):
    # Late import to avoid Keras import until TF bindings are set.
    from .network import load_model

    if volumes is None:
        raise ValueError('Volumes must be provided.')

    if bounds_input_file is not None:
        gen_kwargs = {
                k: {'bounds_generator': iter(SubvolumeBounds.iterable_from_csv(bounds_input_file.format(volume=k)))}
                for k in volumes.iterkeys()}
    else:
        if moves is None:
            moves = 5
        else:
            moves = np.asarray(moves)
        subv_shape = CONFIG.model.input_fov_shape + CONFIG.model.move_step * 2 * moves

        if sparse:
            gen_kwargs = {
                    k: {'sparse_margin': subv_shape}
                    for k in volumes.iterkeys()}
        else:
            gen_kwargs = {
                    k: {'shape': subv_shape}
                    for k in volumes.iterkeys()}
    regions = Roundrobin(*[
            Region.from_subvolume_generator(
                v.downsample(CONFIG.volume.resolution)
                 .subvolume_generator(**gen_kwargs[k]))
            for k, v in six.iteritems(volumes)])

    model = load_model(model_file, CONFIG.network)

    for region in regions:
        region.bias_against_merge = bias
        region.fill(model,
                    progress=True,
                    move_batch_size=move_batch_size,
                    max_moves=max_moves,
                    remask_interval=remask_interval)
        body = region.to_body()
        mask, bounds = body.get_seeded_component(CONFIG.postprocessing.closing_shape)
        viewer = region.get_viewer()
        viewer.add(mask.astype(np.float32),
                   name='Body Mask',
                   offset=bounds[0],
                   shader=get_color_shader(2))
        print(viewer)
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


def partition_volumes(volumes):
    """Paritition volumes into training and validation based on configuration.

    Uses the regexes mapping partition sizes and indices in
    diluvian.config.TrainingConfig by applying them to matching volumes based
    on name.

    Parameters
    ----------
    volumes : dict
        Dictionary mapping volume name to diluvian.volumes.Volume.

    Returns
    -------
    training_volumes, validation_volumes : dict
        Dictionary mapping volume name to partitioned, downsampled volumes.
    """
    def apply_partitioning(volumes, partitioning):
        partitioned = {}
        for name, vol in six.iteritems(volumes):
            partitions = [p for rgx, p in CONFIG.training.partitions.items() if re.match(rgx, name)]
            partition_index = [idx for rgx, idx in partitioning.items() if re.match(rgx, name)]
            if len(partitions) > 1 or len(partition_index) > 1:
                raise ValueError('Volume "{}" matches more than one partition specifier'.format(name))
            elif len(partitions) == 1 and len(partition_index) == 1:
                partitioned[name] = vol.partition(partitions[0], partition_index[0]) \
                                       .downsample(CONFIG.volume.resolution)

        return partitioned

    training_volumes = apply_partitioning(volumes, CONFIG.training.training_partition)
    validation_volumes = apply_partitioning(volumes, CONFIG.training.validation_partition)

    return training_volumes, validation_volumes


def view_volumes(volumes):
    """Display a set of volumes together in a neuroglancer viewer.

    Parameters
    ----------
    volumes : dict
        Dictionary mapping volume name to diluvian.volumes.Volume.
    """
    viewer = WrappedViewer()

    for volume_name, volume in six.iteritems(volumes):
        viewer.add(volume.image_data,
                   name='{} (Image)'.format(volume_name),
                   voxel_size=list(np.flipud(volume.resolution)))
        if volume.label_data is not None:
            viewer.add(volume.label_data,
                       name='{} (Labels)'.format(volume_name),
                       voxel_size=list(np.flipud(volume.resolution)))
        if volume.mask_data is not None:
            viewer.add(volume.mask_data,
                       name='{} (Mask)'.format(volume_name),
                       voxel_size=list(np.flipud(volume.resolution)))

    viewer.print_view_prompt()
