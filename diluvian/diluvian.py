# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

from collections import deque
import itertools
import logging
from multiprocessing import (
        Lock,
        Process,
        Queue,
        )
import os
import re

import numpy as np
import pytoml as toml
import six
from six.moves import input as raw_input
from tqdm import tqdm

from .config import CONFIG
from . import preprocessing
from .util import (
        Roundrobin,
        WrappedViewer,
        )
from .volumes import (
        HDF5Volume,
        SubvolumeBounds,
        )
from .regions import Region


def generate_subvolume_bounds(filename, volumes, num_bounds, sparse=False):
    if '{volume}' not in filename:
        raise ValueError('CSV filename must contain "{volume}" for volume name replacement.')

    if sparse:
        gen_kwargs = {'sparse_margin': CONFIG.model.training_subv_shape * 4 - 3}
    else:
        gen_kwargs = {'shape': CONFIG.model.training_subv_shape * 4 - 3}
    for k, v in six.iteritems(volumes):
        bounds = v.downsample(CONFIG.volume.resolution)\
                  .subvolume_bounds_generator(**gen_kwargs)
        bounds = itertools.islice(bounds, num_bounds)
        SubvolumeBounds.iterable_to_csv(bounds, filename.format(volume=k))


def fill_subvolume_with_model(
        model_file,
        subvolume,
        seed_generator='sobel',
        background_label_id=0,
        bias=True,
        move_batch_size=1,
        max_bodies=None,
        num_workers=CONFIG.training.num_gpus,
        worker_prequeue=2):
    # Create an output label volume.
    prediction = np.full_like(subvolume.image, background_label_id, dtype=np.uint64)
    # Create a conflict count volume that tracks locations where segmented
    # bodies overlap. For now the first body takes precedence in the
    # predicted labels.
    conflict_count = np.full_like(prediction, 0, dtype=np.uint32)

    def worker(worker_id, model_file, image, seeds, results, lock):
        lock.acquire()
        import tensorflow as tf

        # Only make one GPU visible to Tensorflow so that it does not allocate
        # all available memory on all devices.
        # See: https://stackoverflow.com/questions/37893755
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_id)

        with tf.device('/gpu:0'.format(worker_id)):
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

            logging.debug('Worker %s: got seed %s', worker_id, np.array_str(seed))

            # Flood-fill and get resulting mask.
            # Allow reading outside the image volume bounds to allow segmentation
            # to fill all the way to the boundary.
            region = Region(image, seed_vox=seed, sparse_mask=True, block_padding='reflect')
            region.bias_against_merge = bias
            region.fill(model,
                        move_batch_size=move_batch_size,
                        progress=1 + worker_id)
            body = region.to_body()
            logging.debug('Worker %s: seed %s filled', worker_id, np.array_str(seed))

            results.put((seed, body))

        seed_queue.close()
        results_queue.close()

    # Generate seeds from volume.
    generator = preprocessing.SEED_GENERATORS[seed_generator]
    seeds = generator(subvolume.image)

    pbar = tqdm(desc='Seed queue', total=len(seeds), miniters=1, smoothing=0.0)
    num_seeds = len(seeds)
    seeds = iter(seeds)
    seed_queue = Queue()
    results_queue = Queue()
    dispatched_seeds = deque()
    unordered_results = {}
    for seed in itertools.islice(seeds, min(num_seeds, num_workers * worker_prequeue)):
        dispatched_seeds.append(seed)
        seed_queue.put(seed)

    workers = []
    loading_lock = Lock()
    for worker_id in range(num_workers):
        w = Process(target=worker, args=(worker_id, model_file, subvolume.image,
                                         seed_queue, results_queue, loading_lock))
        w.start()
        workers.append(w)

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

    label_id = 0

    # For each seed, create region, fill, threshold, and merge to output volume.
    while dispatched_seeds:
        expected_seed = dispatched_seeds.popleft()
        logging.debug('Expecting seed %s', np.array_str(expected_seed))
        processed_seeds = 1

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
            continue

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
    seed_queue.close()
    results_queue.close()
    for wid, worker in enumerate(workers):
        worker.join()

    return prediction, conflict_count


def fill_volumes_with_model(
        model_file,
        volumes,
        filename,
        viewer=False,
        **kwargs):
    if '{volume}' not in filename:
        raise ValueError('HDF5 filename must contain "{volume}" for volume name replacement.')

    for volume_name, volume in six.iteritems(volumes):
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
    # Late import to avoid Keras import until TF bindings are set.
    from .network import load_model

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
                    multi_gpu_pad_kludge=multi_gpu_model_kludge)
        viewer = region.get_viewer()
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
        viewer.add(volume.label_data,
                   name='{} (Labels)'.format(volume_name),
                   voxel_size=list(np.flipud(volume.resolution)))
        if volume.mask_data is not None:
            viewer.add(volume.mask_data,
                       name='{} (Mask)'.format(volume_name),
                       voxel_size=list(np.flipud(volume.resolution)))

    viewer.print_view_prompt()
