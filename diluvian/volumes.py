# -*- coding: utf-8 -*-


import h5py
import math
import numpy as np
from PIL import Image
import pytoml as toml
import requests

from keras.utils.data_utils import get_file

from .config import CONFIG
from .octrees import OctreeMatrix
from .regions import DenseRegion
from .util import pad_dims


class Volume(object):
    def __init__(self, image_data, label_data, resolution):
        self.image_data = image_data
        self.label_data = label_data
        self.image_data.flags.writeable = False
        self.label_data.flags.writeable = False
        self.resolution = resolution

    def xyz_coord_to_local(self, a):
        return a

    def xyz_mat_to_local(self, m):
        return m

    def simple_training_generator(self, subvolume_size, batch_size, training_size, f_a_bins=None, partition=None):
        subvolumes = self.SubvolumeGenerator(self, subvolume_size, CONFIG.volume.downsample, partition)

        mask_input = np.full(np.append(subvolume_size, (1,)), CONFIG.model.v_false, dtype='float32')
        mask_input[tuple(np.array(mask_input.shape) / 2)] = CONFIG.model.v_true
        mask_input = np.tile(mask_input, (batch_size, 1, 1, 1, 1))

        if f_a_bins is not None:
            f_a_counts = np.zeros_like(f_a_bins, dtype='uint64')
        f_as = np.zeros(batch_size)

        sample_num = 0
        while 1:
            if sample_num >= training_size:
                subvolumes.reset()
                sample_num = 0

            batch_image_input = [None] * batch_size
            batch_mask_target = [None] * batch_size

            for batch_ind in range(0, batch_size):
                subvolume = subvolumes.next()

                batch_image_input[batch_ind] = pad_dims(subvolume['image'])
                batch_mask_target[batch_ind] = pad_dims(subvolume['mask_target'])
                f_as[batch_ind] = subvolume['f_a']

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
                f_a_counts[inds] += counts.astype('uint64')
                sample_weights = np.reciprocal(f_a_counts[f_a_inds], dtype='float64')
                yield ({'image_input': batch_image_input,
                        'mask_input': mask_input},
                       [batch_mask_target],
                       sample_weights)

    def moving_training_generator(self, subvolume_size, batch_size, training_size, callback_kludge,
                                  f_a_bins=None, partition=None, verbose=False):
        subvolumes = self.SubvolumeGenerator(self, subvolume_size, CONFIG.volume.downsample, partition)

        regions = [None] * batch_size
        region_pos = [None] * batch_size
        move_counts = [0] * batch_size
        epoch_move_counts = []

        if f_a_bins is not None:
            f_a_counts = np.zeros_like(f_a_bins, dtype='uint64')
        f_as = np.zeros(batch_size)

        sample_num = 0
        while 1:
            if sample_num >= training_size:
                subvolumes.reset()
                if verbose and len(epoch_move_counts):
                    print ' Average moves: {}'.format(sum(epoch_move_counts)/float(len(epoch_move_counts)))
                epoch_move_counts = []
                sample_num = 0

            # Before clearing last batches, reuse them to predict mask outputs
            # for move training. Add mask outputs to regions.
            active_regions = [n for n, region in enumerate(regions) if region is not None]
            if active_regions and callback_kludge['outputs'] is not None:
                for n in active_regions:
                    assert np.array_equal(callback_kludge['inputs']['image_input'][n, 0, 0, :, 0], batch_image_input[n, 0, 0, :, 0])
                    regions[n].add_mask(callback_kludge['outputs'][n, :, :, :, 0], region_pos[n])

            batch_image_input = [None] * batch_size
            batch_mask_input = [None] * batch_size
            batch_mask_target = [None] * batch_size

            for r, region in enumerate(regions):
                if region is None or region.queue.empty():
                    subvolume = subvolumes.next()

                    regions[r] = DenseRegion(subvolume['image'], subvolume['mask_target'])
                    region = regions[r]
                    epoch_move_counts.append(move_counts[r])
                    move_counts[r] = 0
                else:
                    move_counts[r] += 1

                block_data = region.get_next_block()

                batch_image_input[r] = pad_dims(block_data['image'])
                batch_mask_input[r] = pad_dims(block_data['mask'])
                batch_mask_target[r] = pad_dims(block_data['target'])
                region_pos[r] = block_data['position']
                f_as[r] = subvolume['f_a']

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
                f_a_counts[inds] += counts.astype('uint64')
                sample_weights = np.reciprocal(f_a_counts[f_a_inds], dtype='float64')
                yield (inputs,
                       [batch_mask_target],
                       sample_weights)

    def region_generator(self, subvolume_size, partition=None, seed_margin=None):
        subvolumes = self.SubvolumeGenerator(self, subvolume_size, CONFIG.volume.downsample, partition)

        if seed_margin is None:
            seed_margin = 10.0

        margin = np.ceil(np.reciprocal(np.array(CONFIG.volume.resolution), dtype='float64') * seed_margin).astype('uint64')

        while 1:
            subvolume = subvolumes.next()
            mask_target = subvolume['mask_target']
            ctr = np.array(mask_target.shape, dtype='uint64') / 2
            seed_fov = (ctr - margin, ctr + margin + 1)
            seed_region = mask_target[seed_fov[0][0]:seed_fov[1][0],
                                      seed_fov[0][1]:seed_fov[1][1],
                                      seed_fov[0][2]:seed_fov[1][2]]
            if not np.unique(seed_region).size == 1:
                print 'Rejecting region with seed margin too small.'
                continue
            region = DenseRegion(subvolume['image'], mask_target)
            yield region

    def sparse_region_generator(self, partition=None, seed_margin=None):
        subvolumes = self.SparseSubvolumeGenerator(self, CONFIG.volume.downsample, partition)

        if seed_margin is None:
            seed_margin = 10.0

        margin = np.ceil(np.reciprocal(np.array(CONFIG.volume.resolution), dtype='float64') * seed_margin).astype('uint64')

        while 1:
            subvolume = subvolumes.next()
            mask_target = subvolume['mask_target']
            ctr = np.array(subvolume['seed'])
            seed_fov = (ctr - margin, ctr + margin + 1)
            seed_region = mask_target[seed_fov[0][0]:seed_fov[1][0],
                                      seed_fov[0][1]:seed_fov[1][1],
                                      seed_fov[0][2]:seed_fov[1][2]]
            if not np.unique(seed_region).size == 1:
                print 'Rejecting region with seed margin too small.'
                continue
            output_mask = OctreeMatrix([64, 64, 24], subvolumes.volume_shape_orig, 'float32')
            output_mask[subvolumes.volume_shape_orig[0][0]:subvolumes.volume_shape_orig[1][0],
                        subvolumes.volume_shape_orig[0][1]:subvolumes.volume_shape_orig[1][1],
                        subvolumes.volume_shape_orig[0][2]:subvolumes.volume_shape_orig[1][2]] = np.NAN
            region = DenseRegion(subvolume['image'],
                                 target=mask_target,
                                 seed_pos=np.floor_divide(ctr, (CONFIG.model.block_size - 1) / 4),
                                 mask=output_mask)
            yield region


    class SubvolumeGenerator(object):
        def __init__(self, volume, size_zoom, downsample, partition=None):
            if partition is None:
                partition = (np.array((1, 1, 1)), np.array((0, 0, 0)))
            self.volume = volume
            self.partition = partition
            self.zoom = np.exp2(downsample).astype('uint64')
            self.size_zoom = size_zoom
            self.size_orig = np.multiply(self.size_zoom, self.zoom).astype('uint64')
            self.margin = np.floor_divide(self.size_orig, 2)
            self.partition_size = np.floor_divide(self.volume.xyz_coord_to_local(np.array(self.volume.image_data.shape)), self.partition[0])
            self.ctr_min = (np.multiply(self.partition_size, self.partition[1]) + self.margin).astype('uint64')
            self.ctr_max = (np.multiply(self.partition_size, self.partition[1] + 1) - self.margin - 1).astype('uint64')
            self.random = np.random.RandomState(0)

        def __iter__(self):
            return self

        def reset(self):
            self.random.seed(0)

        def next(self):
            ctr = np.array([self.random.randint(self.ctr_min[n], self.ctr_max[n]) for n in range(3)]).astype('uint64')
            subvol = (self.volume.xyz_coord_to_local(ctr - self.margin),
                      self.volume.xyz_coord_to_local(ctr + self.margin + np.mod(self.size_orig, 2)))
            image_subvol = self.volume.image_data[
                    subvol[0][0]:subvol[1][0],
                    subvol[0][1]:subvol[1][1],
                    subvol[0][2]:subvol[1][2]]
            label_subvol = self.volume.label_data[
                    subvol[0][0]:subvol[1][0],
                    subvol[0][1]:subvol[1][1],
                    subvol[0][2]:subvol[1][2]]

            image_subvol = self.volume.xyz_mat_to_local(image_subvol.astype('float32')) / 256.0
            label_subvol = self.volume.xyz_mat_to_local(label_subvol)
            label_id = label_subvol[tuple(np.array(label_subvol.shape) / 2)]
            label_mask = label_subvol == label_id

            if np.any(self.zoom > 1):
                image_subvol = image_subvol.reshape([self.size_zoom[0], self.zoom[0],
                                                     self.size_zoom[1], self.zoom[1],
                                                     self.size_zoom[2], self.zoom[2]]).mean(5).mean(3).mean(1)
                label_mask = label_mask.reshape([self.size_zoom[0], self.zoom[0],
                                                 self.size_zoom[1], self.zoom[1],
                                                 self.size_zoom[2], self.zoom[2]]).all(5).all(3).all(1)
                # A higher fidelity alternative would be to use the mode label
                # for each downsample block. However, this is prohibitively
                # slow using the scipy code preserved below as an example:
                # label_mask = label_mask.reshape([self.size_zoom[0], self.zoom[0],
                #                                  self.size_zoom[1], self.zoom[1],
                #                                  self.size_zoom[2], self.zoom[2]])
                # label_mask = stats.mode(label_mask, 5)[0]
                # label_mask = stats.mode(label_mask, 3)[0]
                # label_mask = np.squeeze(stats.mode(label_mask, 1)[0])

            assert image_subvol.shape == tuple(self.size_zoom), 'Image wrong size: {}'.format(image_subvol.shape)
            assert label_mask.shape == tuple(self.size_zoom), 'Labels wrong size: {}'.format(label_mask.shape)

            f_a = np.count_nonzero(label_mask) / float(label_mask.size)
            mask_target = np.full_like(label_mask, CONFIG.model.v_false, dtype='float32')
            mask_target[label_mask] = CONFIG.model.v_true

            return {'image': image_subvol, 'mask_target': mask_target, 'f_a': f_a}


    class SparseSubvolumeGenerator(SubvolumeGenerator):
        def __init__(self, volume, downsample, partition=None):
            super(HDF5Volume.SparseSubvolumeGenerator, self).__init__(volume, CONFIG.model.block_size, downsample, partition)
            self.volume_shape_orig = (np.zeros((3,), dtype='uint64'),
                                      np.divide(self.volume.xyz_coord_to_local(np.array(self.volume.image_data.shape)), self.zoom))

        def next(self):
            ctr = np.array([self.random.randint(self.ctr_min[n], self.ctr_max[n]) for n in range(3)]).astype('uint64')
            label_id = self.volume.label_data[tuple(self.volume.xyz_coord_to_local(ctr))]

            def image_populator(bounds):
                size = bounds[1] - bounds[0]
                subvol = (self.volume.xyz_coord_to_local(np.multiply(bounds[0], self.zoom)),
                          self.volume.xyz_coord_to_local(np.multiply(bounds[1], self.zoom)))
                image_subvol = self.volume.image_data[
                        subvol[0][0]:subvol[1][0],
                        subvol[0][1]:subvol[1][1],
                        subvol[0][2]:subvol[1][2]]

                image_subvol = self.volume.xyz_mat_to_local(image_subvol.astype('float32')) / 256.0

                if np.any(self.zoom > 1):
                    image_subvol = image_subvol.reshape([size[0], self.zoom[0],
                                                         size[1], self.zoom[1],
                                                         size[2], self.zoom[2]]).mean(5).mean(3).mean(1)
                return image_subvol

            def label_populator(bounds):
                size = bounds[1] - bounds[0]
                subvol = (self.volume.xyz_coord_to_local(np.multiply(bounds[0], self.zoom)),
                          self.volume.xyz_coord_to_local(np.multiply(bounds[1], self.zoom)))
                label_subvol = self.volume.label_data[
                        subvol[0][0]:subvol[1][0],
                        subvol[0][1]:subvol[1][1],
                        subvol[0][2]:subvol[1][2]]

                label_subvol = self.volume.xyz_mat_to_local(label_subvol)
                label_mask = label_subvol == label_id

                if np.any(self.zoom > 1):
                    label_mask = label_mask.reshape([size[0], self.zoom[0],
                                                     size[1], self.zoom[1],
                                                     size[2], self.zoom[2]]).all(5).all(3).all(1)
                mask_target = np.full_like(label_mask, CONFIG.model.v_false, dtype='float32')
                mask_target[label_mask] = CONFIG.model.v_true
                return mask_target

            image_tree = OctreeMatrix([64, 64, 24], self.volume_shape_orig, 'float32', populator=image_populator)
            target_tree = OctreeMatrix([64, 64, 24], self.volume_shape_orig, 'float32', populator=label_populator)

            f_a = 0.0

            return {'image': image_tree, 'mask_target': target_tree, 'f_a': f_a, 'seed': np.divide(ctr, self.zoom)}


class HDF5Volume(Volume):
    @staticmethod
    def from_toml(filename):
        volumes = {}
        with open(filename, 'rb') as fin:
            datasets = toml.load(fin).get('dataset', [])
            for dataset in datasets:
                hdf5_file = dataset['hdf5_file']
                if dataset.get('use_keras_cache', False):
                    hdf5_file = get_file(hdf5_file, dataset['download_url'], md5_hash=dataset.get('download_md5', None))
                volumes[dataset['name']] = HDF5Volume(hdf5_file,
                                                      dataset['image_dataset'],
                                                      dataset['label_dataset'])

        return volumes

    def __init__(self, orig_file, image_dataset, label_dataset):
        self.file = h5py.File(orig_file, 'r')
        self.image_data = self.file[image_dataset]
        self.label_data = self.file[label_dataset]
        if 'resolution' in self.file[image_dataset].attrs:
            self.resolution = np.array(self.file[image_dataset].attrs['resolution'])
        else:
            self.resolution = np.ones(3)

    def xyz_coord_to_local(self, a):
        return np.flipud(a)

    def xyz_mat_to_local(self, m):
        return np.transpose(m)

    def to_memory_volume(self):
        return Volume(self.xyz_mat_to_local(self.image_data[:, :, :]),
                      self.xyz_mat_to_local(self.label_data[:, :, :]),
                      self.xyz_coord_to_local(self.resolution))


class ImageStackVolume(Volume):
    @staticmethod
    def from_catmaid_stack(stack_info, tile_source_parameters):
        # See https://catmaid.readthedocs.io/en/stable/tile_sources.html
        format_url = {
            1: '{source_base_url}{{z}}/{{row}}_{{col}}_{{zoom_level}}.{file_extension}',
            4: '{source_base_url}{{z}}/{{zoom_level}}/{{row}}_{{col}}.{file_extension}',
            5: '{source_base_url}{{zoom_level}}/{{z}}/{{row}}/{{col}}.{file_extension}',
            7: '{source_base_url}largeDataTileSource/{tile_width}/{tile_height}/{{zoom_level}}/{{z}}/{{row}}/{{col}}.{file_extension}',
            9: '{source_base_url}{{z}}/{{row}}_{{col}}_{{zoom_level}}.{file_extension}',
        }[tile_source_parameters['tile_source_type']].format(**tile_source_parameters)
        bounds = np.array(stack_info['bounds'], dtype='uint64')
        resolution = np.array(stack_info['resolution'])
        tile_width = int(tile_source_parameters['tile_width'])
        tile_height = int(tile_source_parameters['tile_height'])
        return ImageStackVolume(bounds, resolution, tile_width, tile_height, format_url)

    def __init__(self, bounds, resolution, tile_width, tile_height, tile_format_url):
        self.resolution = resolution
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tile_format_url = tile_format_url
        self.zoom_level = min(CONFIG.volume.downsample[0], CONFIG.volume.downsample[1])
        scale = np.exp2(np.array([self.zoom_level, self.zoom_level, 0])).astype('uint64')

        data_size = (np.zeros(3), np.divide(bounds, scale).astype('uint64'))
        self.image_data = OctreeMatrix([512, 512, 10],
                                       data_size,
                                       'uint8',
                                       populator=self.image_populator)

        self.label_data = OctreeMatrix([64, 64, 24], data_size, 'uint64')
        self.label_data[data_size[0][0]:data_size[1][0],
                        data_size[0][1]:data_size[1][1],
                        data_size[0][2]:data_size[1][2]] = 1

    def image_populator(self, bounds):
        image_subvol = np.zeros(tuple(bounds[1] - bounds[0]), dtype='uint8')
        col_range = map(int, (math.floor(bounds[0][0]/self.tile_width), math.ceil(bounds[1][0]/self.tile_width)))
        row_range = map(int, (math.floor(bounds[0][1]/self.tile_height), math.ceil(bounds[1][1]/self.tile_height)))
        tile_size = np.array([self.tile_width, self.tile_height, 1]).astype('int64')
        for z in xrange(bounds[0][2], bounds[1][2]):
            for r in xrange(*row_range):
                for c in xrange(*col_range):
                    url = self.tile_format_url.format(zoom_level=self.zoom_level, z=z, row=r, col=c)
                    try:
                        im = np.transpose(np.array(Image.open(requests.get(url, stream=True).raw)))
                    except IOError:
                        im = np.full((self.tile_width, self.tile_height), 0, dtype='uint8')
                    tile_coord = np.array([c, r, z]).astype('int64')
                    tile_loc = np.multiply(tile_coord, tile_size)

                    subvol = (np.maximum(np.zeros(3), tile_loc - bounds[0]).astype('uint64'),
                              np.minimum(np.array(image_subvol.shape), tile_loc + tile_size - bounds[0]).astype('uint64'))
                    tile_sub = (np.maximum(np.zeros(3), bounds[0] - tile_loc).astype('uint64'),
                                np.minimum(tile_size, bounds[1] - tile_loc).astype('uint64'))

                    image_subvol[subvol[0][0]:subvol[1][0],
                                 subvol[0][1]:subvol[1][1],
                                 subvol[0][2]             ] = im[tile_sub[0][0]:tile_sub[1][0],
                                                                 tile_sub[0][1]:tile_sub[1][1]]

        return image_subvol

    class SubvolumeGenerator(Volume.SubvolumeGenerator):
        def __init__(self, volume, size_zoom, downsample, partition=None):
            adjusted_downsample = downsample.copy()
            if volume.zoom_level is not None:
                adjusted_downsample[0:2] -= volume.zoom_level
            super(ImageStackVolume.SubvolumeGenerator, self).__init__(volume, size_zoom, adjusted_downsample, partition)

    class SparseSubvolumeGenerator(Volume.SubvolumeGenerator):
        def __init__(self, volume, downsample, partition=None):
            adjusted_downsample = downsample.copy()
            if volume.zoom_level is not None:
                adjusted_downsample[0:2] -= volume.zoom_level
            super(ImageStackVolume.SparseSubvolumeGenerator, self).__init__(volume, CONFIG.model.block_size, adjusted_downsample, partition)
            self.volume_shape_orig = (np.zeros(3, dtype='uint64'), np.array(self.volume.image_data.shape))


        def next(self):
            ctr = np.array([self.random.randint(self.ctr_min[n], self.ctr_max[n]) for n in range(3)]).astype('uint64')

            f_a = 0.0

            return {'image': self.volume.image_data, 'mask_target': self.volume.label_data, 'f_a': f_a, 'seed': ctr}
