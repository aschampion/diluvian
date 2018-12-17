# -*- coding: utf-8 -*-
"""Volumes of raw image and labeled object data."""


from __future__ import division

from collections import namedtuple
import csv
import logging
import os
import re

import h5py
import math
import numpy as np
from PIL import Image
import pytoml as toml
import requests
from scipy import ndimage
import six
from six.moves import range as xrange

from .config import CONFIG
from .octrees import OctreeVolume
from .util import get_nonzero_aabb


DimOrder = namedtuple('DimOrder', ('X', 'Y', 'Z'))


def partition_volumes(volumes, downsample=True):
    """Paritition volumes into training and validation based on configuration.

    Uses the regexes mapping partition sizes and indices in
    diluvian.config.TrainingConfig by applying them to matching volumes based
    on name.

    Parameters
    ----------
    volumes : dict
        Dictionary mapping volume name to diluvian.volumes.Volume.
    downsample : bool, optional
        Whether to downsample partitions automatically.

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
                v = vol.partition(partitions[0], partition_index[0])
                if downsample:
                    v = v.downsample(CONFIG.volume.resolution)
                partitioned[name] = v

        return partitioned

    training_volumes = apply_partitioning(volumes, CONFIG.training.training_partition)
    validation_volumes = apply_partitioning(volumes, CONFIG.training.validation_partition)

    return training_volumes, validation_volumes


class SubvolumeBounds(object):
    """Sufficient parameters to extract a subvolume from a volume."""
    __slots__ = ('start', 'stop', 'seed', 'label_id', 'label_margin',)

    def __init__(self, start=None, stop=None, seed=None, label_id=None, label_margin=None):
        assert (start is not None and stop is not None) or seed is not None, "Bounds or seed must be provided"
        self.start = start
        self.stop = stop
        self.seed = seed
        self.label_id = label_id
        if label_margin is None:
            label_margin = np.zeros(3, dtype=np.int64)
        self.label_margin = label_margin

    @classmethod
    def iterable_from_csv(cls, filename):
        bounds = []
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for k, v in six.iteritems(row):
                    if not v:
                        row[k] = None
                    elif v[0] == '[':
                        row[k] = np.fromstring(v[1:-1], sep=' ', dtype=np.int64)
                    else:
                        row[k] = int(v)
                bounds.append(cls(**row))

        return bounds

    @classmethod
    def iterable_to_csv(cls, bounds, filename):
        with open(filename, 'w') as csvfile:
            fieldnames = cls.__slots__
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for bound in bounds:
                writer.writerow([getattr(bound, f) for f in fieldnames])


class Subvolume(object):
    """A subvolume of image data and an optional ground truth object mask."""
    __slots__ = ('image', 'label_mask', 'seed', 'label_id',)

    def __init__(self, image, label_mask, seed, label_id):
        self.image = image
        self.label_mask = label_mask
        self.seed = seed
        self.label_id = label_id

    def f_a(self):
        """Calculate the mask filling fraction of this subvolume.

        Returns
        -------
        float
            Fraction of the subvolume voxels in the object mask.
        """
        return np.count_nonzero(self.label_mask) / float(self.label_mask.size)

    def has_seed_in_mask(self):
        ctr = self.seed - (np.asarray(self.image.shape) - np.asarray(self.label_mask.shape)) // 2
        return self.label_mask[tuple(ctr)]

    def has_uniform_seed_margin(self, seed_margin=20.0):
        """Test if a subvolume has a margin of uniform label around its seed.

        Parameters
        ----------
        seed_margin : float, optional
            The minimum acceptable margin of uniform target label around the seed
            voxel (in nm, default 20.0).

        Returns
        -------
        bool
            True if the rectangular margin around the seed position is uniform.
        """
        margin = np.ceil(np.reciprocal(np.array(CONFIG.volume.resolution),
                                       dtype=np.float64) * seed_margin).astype(np.int64)

        mask_target = self.label_mask
        # If data is unlabeled, can not test so always succeed.
        if mask_target is None:
            return True
        # Seed location in the mask accounting for offset of label from image.
        ctr = self.seed - (np.asarray(self.image.shape) - np.asarray(mask_target.shape)) // 2
        seed_fov = (ctr - margin, ctr + margin + 1)
        seed_region = mask_target[seed_fov[0][0]:seed_fov[1][0],
                                  seed_fov[0][1]:seed_fov[1][1],
                                  seed_fov[0][2]:seed_fov[1][2]]
        return np.all(seed_region)


class SubvolumeGenerator(six.Iterator):
    """Combines a volume and a subvolume bounds generator into a generator.

    Parameters
    ----------
    volume : Volume
    bounds_generator : SubvolumeBoundsGenerator
    """
    def __init__(self, volume, bounds_generator):
        self.volume = volume
        self.bounds_generator = bounds_generator

    @property
    def shape(self):
        return self.bounds_generator.shape

    def __iter__(self):
        return self

    def reset(self):
        self.bounds_generator.reset()

    def __next__(self):
        return self.volume.get_subvolume(six.next(self.bounds_generator))


class ErodedMaskGenerator(six.Iterator):
    def __init__(self, subvolume_generator, erosion_px):
        self.subvolume_generator = subvolume_generator
        self.sel = np.ones(erosion_px * 2 + 1)

    @property
    def shape(self):
        return self.subvolume_generator.shape

    def __iter__(self):
        return self

    def reset(self):
        self.subvolume_generator.reset()

    def __next__(self):
        while True:
            subv = six.next(self.subvolume_generator)

            subv.label_mask = ndimage.binary_erosion(subv.label_mask, structure=self.sel, border_value=1)

            if subv.has_seed_in_mask():
                return subv


class RelabelSeedComponentGenerator(six.Iterator):
    def __init__(self, subvolume_generator):
        self.subvolume_generator = subvolume_generator

    @property
    def shape(self):
        return self.subvolume_generator.shape

    def __iter__(self):
        return self

    def reset(self):
        self.subvolume_generator.reset()

    def __next__(self):
        subv = six.next(self.subvolume_generator)

        label_im, _ = ndimage.label(subv.label_mask)
        label_axis_margin = (np.array(subv.image.shape) - np.array(subv.label_mask.shape)) // 2
        seed_label = label_im[tuple(subv.seed - label_axis_margin)]

        subv.label_mask = label_im == seed_label

        return subv


class SubvolumeAugmentGenerator(six.Iterator):
    """Base class for subvolume generator augmenters.

    Parameters
    ----------
    subvolume_generator : SubvolumeGenerator
    return_both : bool
        If true, return both the original and augmented volume in sequence.
        If false, return either with equal probability.
    """
    def __init__(self, subvolume_generator, return_both):
        self.subvolume_generator = subvolume_generator
        self.return_both = return_both
        self.return_single_p = 0.5
        self.subvolume = None

    @property
    def shape(self):
        return self.subvolume_generator.shape

    def __iter__(self):
        return self

    def reset(self):
        self.subvolume = None
        self.subvolume_generator.reset()

    def __next__(self):
        if self.return_both:
            if self.subvolume is None:
                self.subvolume = six.next(self.subvolume_generator)
                return self.subvolume
            else:
                subv = self.augment_subvolume()
                self.subvolume = None
                if subv is None:
                    return six.next(self)
                else:
                    return subv
        else:
            self.subvolume = six.next(self.subvolume_generator)
            if np.random.sample() < self.return_single_p:
                return self.subvolume
            else:
                subv = self.augment_subvolume()
                if subv is None:
                    return self.subvolume
                else:
                    return subv

    def augment_subvolume(self):
        raise NotImplementedError('Subclasses must implement this method.')


class ClipSubvolumeImageGenerator(six.Iterator):
    """Clip subvolume image range (default between zero and one).

    Useful to apply after a sequence of augmentations.

    Parameters
    ----------
    subvolume_generator : SubvolumeGenerator
    min_val, max_val : float, optional
    """
    def __init__(self, subvolume_generator, min_val=0.0, max_val=1.0):
        self.subvolume_generator = subvolume_generator
        self.min_val = min_val
        self.max_val = max_val

    @property
    def shape(self):
        return self.subvolume_generator.shape

    def __iter__(self):
        return self

    def reset(self):
        self.subvolume_generator.reset()

    def __next__(self):
        subv = six.next(self.subvolume_generator)
        return Subvolume(np.clip(subv.image, self.min_val, self.max_val),
                         subv.label_mask,
                         subv.seed,
                         subv.label_id)


class MirrorAugmentGenerator(SubvolumeAugmentGenerator):
    """Repeats subvolumes from a subvolume generator mirrored along an axis.

    For each subvolume in the original generator, this generator will yield two
    subvolumes: the original subvolume and the subvolume with the image,
    label mask, and seed mirrored along a given axis.

    Parameters
    ----------
    subvolume_generator : SubvolumeGenerator
    return_both : bool
        If true, return both the original and augmented volume in sequence.
        If false, return either with equal probability.
    axis : int
    """
    def __init__(self, subvolume_generator, return_both, axis):
        super(MirrorAugmentGenerator, self).__init__(subvolume_generator, return_both)
        self.axis = axis

    def augment_subvolume(self):
        subv = self.subvolume
        shape = subv.image.shape[self.axis]
        seed = subv.seed.copy()
        seed[self.axis] = shape - subv.seed[self.axis] - 1
        subv = Subvolume(np.flip(subv.image, self.axis),
                         np.flip(subv.label_mask, self.axis) if subv.label_mask is not None else None,
                         seed,
                         subv.label_id)
        return subv


class PermuteAxesAugmentGenerator(SubvolumeAugmentGenerator):
    """Repeats subvolumes from a subvolume generator with an axes permutation.

    For each subvolume in the original generator, this generator will yield two
    subvolumes: the original subvolume and the subvolume with the image,
    label mask, and seed axes permuted according to a given axes order.

    Parameters
    ----------
    subvolume_generator : SubvolumeGenerator
    return_both : bool
        If true, return both the original and augmented volume in sequence.
        If false, return either with equal probability.
    axes : sequence of int
    """
    def __init__(self, subvolume_generator, return_both, axes):
        super(PermuteAxesAugmentGenerator, self).__init__(subvolume_generator, return_both)
        self.axes = list(axes)

    def augment_subvolume(self):
        subv = self.subvolume
        subv = Subvolume(np.transpose(subv.image, self.axes),
                         np.transpose(subv.label_mask, self.axes) if subv.label_mask is not None else None,
                         subv.seed[self.axes],
                         self.subvolume.label_id)
        return subv


class MissingDataAugmentGenerator(SubvolumeAugmentGenerator):
    """Repeats subvolumes from a subvolume generator with missing data planes.

    For each subvolume in the original generator, this generator will yield the
    original subvolume and may yield a subvolume with missing planes of image
    and/or label mask data.

    Parameters
    ----------
    subvolume_generator : SubvolumeGenerator
    return_both : bool
        If true, return both the original and augmented volume in sequence.
        If false, return either with equal probability.
    axis : int
    probability : float
        Independent probability that each plane of data along axis is missing.
    remove_label : bool
        Whether to also remove label mask data.
    """
    def __init__(self, subvolume_generator, return_both, axis, probability, remove_label=False):
        super(MissingDataAugmentGenerator, self).__init__(subvolume_generator, return_both)
        self.axis = axis
        self.probability = probability
        self.remove_label = remove_label

    def augment_subvolume(self):
        rolls = np.random.sample(self.shape[self.axis])
        # Remove the seed plane from possibilities.
        rolls[self.subvolume.seed[self.axis]] = 1.1
        missing_sections = np.where(rolls < self.probability)

        if missing_sections and missing_sections[0].size:
            subv = self.subvolume
            mask = subv.label_mask.copy() if subv.label_mask is not None and self.remove_label else subv.label_mask
            subv = Subvolume(subv.image.copy(),
                             mask,
                             subv.seed,
                             subv.label_id)
            slices = [slice(None), slice(None), slice(None)]
            slices[self.axis] = missing_sections
            subv.image[slices] = 0
            if self.remove_label:
                label_axis_margin = (subv.image.shape[self.axis] - subv.label_mask.shape[self.axis]) // 2
                label_sections = missing_sections[0] - label_axis_margin
                label_sections = label_sections[(label_sections >= 0) &
                                                (label_sections < subv.label_mask.shape[self.axis])]
                slices[self.axis] = (label_sections,)
                subv.label_mask[slices] = False
            return subv
        else:
            # No augmentations to be made. Superclass will automatically return
            # next subvolume.
            return None


class GaussianNoiseAugmentGenerator(SubvolumeAugmentGenerator):
    """Repeats subvolumes from a subvolume generator with Gaussian noise.

    For each subvolume in the original generator, this generator will yield two
    subvolumes: the original subvolume and the subvolume with multiplicative
    and additive Gaussian noise applied to the image data.

    Parameters
    ----------
    subvolume_generator : SubvolumeGenerator
    return_both : bool
        If true, return both the original and augmented volume in sequence.
        If false, return either with equal probability.
    axis : int
        Axis along which noise will be applied independently. For example,
        0 will apply different noise to each z-section. -1 will apply
        uniform noise to the entire subvolume.
    multiplicative : float
        Standard deviation for 1-mean Gaussian multiplicative noise.
    multiplicative : float
        Standard deviation for 0-mean Gaussian additive noise.
    """
    def __init__(self, subvolume_generator, return_both, axis, multiplicative, additive):
        super(GaussianNoiseAugmentGenerator, self).__init__(subvolume_generator, return_both)
        self.axis = axis
        self.multiplicative = multiplicative
        self.additive = additive

    def augment_subvolume(self):
        subv = self.subvolume

        # Generate a transformed shape that will apply vector addition
        # and multiplication along to correct axis.
        shape_xform = np.ones((1, 3), dtype=np.int32).ravel()
        shape_xform[self.axis] = -1

        dim_size = 1 if self.axis == -1 else self.shape[self.axis]
        mul_noise = np.random.normal(1.0, self.multiplicative, dim_size).astype(subv.image.dtype)
        add_noise = np.random.normal(0.0, self.additive, dim_size).astype(subv.image.dtype)

        subv = Subvolume(subv.image * mul_noise.reshape(shape_xform) + add_noise.reshape(shape_xform),
                         subv.label_mask,
                         subv.seed,
                         subv.label_id)
        return subv


class ContrastAugmentGenerator(SubvolumeAugmentGenerator):
    """Repeats subvolumes from a subvolume generator with altered contrast.

    For each subvolume in the original generator, this generator will yield the
    original subvolume and may yield a subvolume with image intensity contrast.

    Currently this augmentation performs simple rescaling of intensity values,
    not histogram based methods. This simple approach still yields results
    resembling TEM artifacts. A single rescaling is chosen for all selected
    sections in each subvolume, not independently per selected section.

    Parameters
    ----------
    subvolume_generator : SubvolumeGenerator
    return_both : bool
        If true, return both the original and augmented volume in sequence.
        If false, return either with equal probability.
    axis : int
        Axis along which contrast may be altered. For example, 0 will alter
        contrast by z-sections.
    probability : float
        Independent probability that each plane of data along axis is altered.
    scaling_mean, scaling_std, center_mean, center_std : float
        Normal distribution parameters for the rescaling of intensity values.
    """
    def __init__(self, subvolume_generator, return_both, axis, probability,
                 scaling_mean, scaling_std, center_mean, center_std):
        super(ContrastAugmentGenerator, self).__init__(subvolume_generator, return_both)
        self.axis = axis
        self.probability = probability
        self.scaling_mean = scaling_mean
        self.scaling_std = scaling_std
        self.center_mean = center_mean
        self.center_std = center_std

    def augment_subvolume(self):
        rolls = np.random.sample(self.shape[self.axis])
        sections = np.where(rolls < self.probability)

        if sections and sections[0].size:
            subv = self.subvolume
            subv = Subvolume(subv.image.copy(),
                             subv.label_mask,
                             subv.seed,
                             subv.label_id)
            slices = [slice(None), slice(None), slice(None)]
            slices[self.axis] = sections
            data = subv.image[slices]
            old_min = data.min()
            old_max = data.max()
            scaling = np.random.normal(self.scaling_mean, self.scaling_std)
            center = np.random.normal(self.center_mean, self.center_std)
            data = scaling*(data - old_min) + 0.5*scaling*center*(old_max - old_min) + old_min
            subv.image[slices] = data
            return subv
        else:
            return None


class MaskedArtifactAugmentGenerator(SubvolumeAugmentGenerator):
    """Repeats subvolumes from a subvolume generator with artifact data added.

    For each subvolume in the original generator, this generator will yield the
    original subvolume and may yield a subvolume with planes of image mixed
    with artifact data from a separate volume.

    Parameters
    ----------
    subvolume_generator : SubvolumeGenerator
    return_both : bool
        If true, return both the original and augmented volume in sequence.
        If false, return either with equal probability.
    axis : int
    probability : float
        Independent probability that each plane of data along axis has
        artifacts.
    artifact_volume_file : string
        Filename of an TOML descriptor of an HDF5 dataset with image and mask
        data channels. Only the dataset named 'Artifacts' from this descriptor
        will be used. Mask data should be a float that will be interpreted
        as an alpha for blending image data from this artifact file with
        the original subvolume image data.
    """
    def __init__(self, subvolume_generator, return_both, axis, probability, artifact_volume_file, cache):
        super(MaskedArtifactAugmentGenerator, self).__init__(subvolume_generator, return_both)
        self.axis = axis
        self.probability = probability
        if 'artifacts' not in cache:
            vol = HDF5Volume.from_toml(artifact_volume_file)['Artifacts']
            cache['mask'] = NdarrayVolume(
                    vol.world_coord_to_local(vol.resolution),
                    image_data=vol.world_mat_to_local(vol.mask_data[:]))
            vol.mask_data = None
            cache['artifacts'] = vol.to_memory_volume()
        self.mask = cache['mask']
        self.artifacts = cache['artifacts']
        artifact_shape = self.shape.copy()
        artifact_shape[self.axis] = 1
        self.art_bounds_gen = self.artifacts.subvolume_bounds_generator(shape=artifact_shape)

    def augment_subvolume(self):
        rolls = np.random.sample(self.shape[self.axis])
        artifact_sections = np.where(rolls < self.probability)

        if artifact_sections and artifact_sections[0].size:
            subv = self.subvolume
            subv = Subvolume(subv.image.copy(),
                             subv.label_mask,
                             subv.seed,
                             subv.label_id)
            slices = [slice(None), slice(None), slice(None)]
            for z in artifact_sections[0]:
                slices[self.axis] = z
                mask_found = False
                # Since artifact data is usually sparse, reject patches
                # that have all zero mask.
                while not mask_found:
                    art_bounds = six.next(self.art_bounds_gen)
                    mask = self.mask.get_subvolume(art_bounds).image
                    if mask.max() == 0.0:
                        continue
                    mask_found = True
                    art = self.artifacts.get_subvolume(art_bounds).image
                raw = subv.image[slices]
                subv.image[slices] = raw * (1.0 - mask) + art * mask
            return subv
        else:
            return None


class Volume(object):
    DIM = DimOrder(Z=0, Y=1, X=2)

    def __init__(self, resolution, image_data=None, label_data=None, mask_data=None):
        self.resolution = resolution
        self.image_data = image_data
        self.label_data = label_data
        self.mask_data = mask_data
        self._mask_bounds = None

    def local_coord_to_world(self, a):
        return a

    def world_coord_to_local(self, a):
        return a

    def world_mat_to_local(self, m):
        return m

    @property
    def mask_bounds(self):
        if self._mask_bounds is not None:
            return self._mask_bounds
        if self.mask_data is None:
            return None

        # Explicitly copy the channel to memory. 3x speedup for np ops.
        mask_data = self.mask_data[:]

        self._mask_bounds = get_nonzero_aabb(mask_data)

        return self._mask_bounds

    @property
    def shape(self):
        return tuple(self.world_coord_to_local(np.array(self.image_data.shape)))

    def _get_downsample_from_resolution(self, resolution):
        resolution = np.asarray(resolution)
        downsample = np.log2(np.true_divide(resolution, self.resolution))
        if np.any(downsample < 0):
            raise ValueError('Requested resolution ({}) is higher than volume resolution ({}). '
                             'Upsampling is not supported.'.format(resolution, self.resolution))
        if not np.all(np.equal(np.mod(downsample, 1), 0)):
            raise ValueError('Requested resolution ({}) is not a power-of-2 downsample of '
                             'volume resolution ({}). '
                             'This is currently unsupported.'.format(resolution, self.resolution))
        return downsample.astype(np.int64)

    def downsample(self, resolution):
        downsample = self._get_downsample_from_resolution(resolution)
        if np.all(np.equal(downsample, 0)):
            return self
        return DownsampledVolume(self, downsample)

    def partition(self, partitioning, partition_index):
        if np.array_equal(partitioning, np.ones(3)) and np.array_equal(partition_index, np.zeros(3)):
            return self
        return PartitionedVolume(self, partitioning, partition_index)

    def sparse_wrapper(self, *args):
        return SparseWrappedVolume(self, *args)

    def subvolume_bounds_generator(self, shape=None, label_margin=None):
        return self.SubvolumeBoundsGenerator(self, shape, label_margin)

    def subvolume_generator(self, bounds_generator=None, **kwargs):
        if bounds_generator is None:
            if not kwargs:
                raise ValueError('Bounds generator arguments must be provided if no bounds generator is provided.')
            bounds_generator = self.subvolume_bounds_generator(**kwargs)
        return SubvolumeGenerator(self, bounds_generator)

    def get_subvolume(self, bounds):
        if bounds.start is None or bounds.stop is None:
            raise ValueError('This volume does not support sparse subvolume access.')

        image_subvol = self.image_data[
                bounds.start[0]:bounds.stop[0],
                bounds.start[1]:bounds.stop[1],
                bounds.start[2]:bounds.stop[2]]

        image_subvol = self.world_mat_to_local(image_subvol)
        if np.issubdtype(image_subvol.dtype, np.integer):
            image_subvol = image_subvol.astype(np.float32) / 256.0

        seed = bounds.seed
        if seed is None:
            seed = np.array(image_subvol.shape, dtype=np.int64) // 2

        if self.label_data is not None:
            label_start = bounds.start + bounds.label_margin
            label_stop = bounds.stop - bounds.label_margin

            label_subvol = self.label_data[
                    label_start[0]:label_stop[0],
                    label_start[1]:label_stop[1],
                    label_start[2]:label_stop[2]]

            label_subvol = self.world_mat_to_local(label_subvol)

            label_id = bounds.label_id
            if label_id is None:
                label_id = label_subvol[tuple(seed - bounds.label_margin)]
            label_mask = label_subvol == label_id
        else:
            label_mask = None
            label_id = None

        return Subvolume(image_subvol, label_mask, seed, label_id)

    class SubvolumeBoundsGenerator(six.Iterator):
        def __init__(self, volume, shape, label_margin=None):
            self.volume = volume
            self.shape = shape
            self.margin = np.floor_divide(self.shape, 2).astype(np.int64)
            if label_margin is None:
                label_margin = np.zeros(3, dtype=np.int64)
            self.label_margin = label_margin
            self.skip_blank_sections = True
            self.ctr_min = self.margin
            self.ctr_max = (np.array(self.volume.shape) - self.margin - 1).astype(np.int64)
            self.random = np.random.RandomState(CONFIG.random_seed)

            # If the volume has a mask channel, further limit ctr_min and
            # ctr_max to lie inside a margin in the AABB of the mask.
            if self.volume.mask_data is not None:
                mask_min, mask_max = self.volume.mask_bounds

                mask_min = self.volume.world_coord_to_local(mask_min)
                mask_max = self.volume.world_coord_to_local(mask_max)

                self.ctr_min = np.maximum(self.ctr_min, mask_min + self.label_margin)
                self.ctr_max = np.minimum(self.ctr_max, mask_max - self.label_margin - 1)

            if np.any(self.ctr_min >= self.ctr_max):
                raise ValueError('Cannot generate subvolume bounds: bounds ({}, {}) too small for shape ({})'.format(
                                 np.array_str(self.ctr_min), np.array_str(self.ctr_max), np.array_str(self.shape)))

        def __iter__(self):
            return self

        def reset(self):
            self.random.seed(0)

        def __next__(self):
            while True:
                ctr = np.array([self.random.randint(self.ctr_min[n], self.ctr_max[n])
                                for n in range(3)]).astype(np.int64)
                start = ctr - self.margin
                stop = ctr + self.margin + np.mod(self.shape, 2).astype(np.int64)

                # If the volume has a mask channel, only accept subvolumes
                # entirely contained in it.
                if self.volume.mask_data is not None:
                    start_local = self.volume.world_coord_to_local(start + self.label_margin)
                    stop_local = self.volume.world_coord_to_local(stop - self.label_margin)
                    mask = self.volume.mask_data[
                            start_local[0]:stop_local[0],
                            start_local[1]:stop_local[1],
                            start_local[2]:stop_local[2]]
                    if not mask.all():
                        logging.debug('Skipping subvolume not entirely in mask.')
                        continue

                # Skip subvolumes with seeds in blank sections.
                if self.skip_blank_sections and self.volume.image_data is not None:
                    if self.volume.image_data[tuple(self.volume.local_coord_to_world(ctr))] == 0:
                        logging.debug('Skipping subvolume with seed in blank section.')
                        continue

                # Only accept subvolumes where the central seed voxel will be
                # of a uniform label after downsampling. For more stringent
                # seed region uniformity filtering, see has_uniform_seed_margin.
                if self.volume.label_data is None:
                    label_id = None
                    break
                seed_min = self.volume.local_coord_to_world(ctr)
                seed_max = self.volume.local_coord_to_world(ctr + 1)
                label_ids = self.volume.label_data[
                        seed_min[0]:seed_max[0],
                        seed_min[1]:seed_max[1],
                        seed_min[2]:seed_max[2]]
                if (label_ids == label_ids.item(0)).all():
                    label_id = label_ids.item(0)
                    break

            return SubvolumeBounds(start, stop, label_id=label_id, label_margin=self.label_margin)


class NdarrayVolume(Volume):
    """A NumPy ndarray-backed volume.

    Since all volumes assume image and label data are ndarray-like, this class
    exists mostly as a bookkeeping convenience to make actual ndarray volumes
    explicit.
    """
    def __init__(self, *args, **kwargs):
        super(NdarrayVolume, self).__init__(*args, **kwargs)
        self.image_data.flags.writeable = False
        if self.label_data is not None:
            self.label_data.flags.writeable = False


class VolumeView(Volume):
    def __init__(self, parent, *args, **kwargs):
        super(VolumeView, self).__init__(*args, **kwargs)
        self.parent = parent

    def local_to_parent(self, a):
        return a

    def local_coord_to_world(self, a):
        return self.parent.local_coord_to_world(self.local_to_parent(a))

    def parent_to_local(self, a):
        return a

    def world_coord_to_local(self, a):
        return self.parent_to_local(self.parent.world_coord_to_local(a))

    def world_mat_to_local(self, m):
        return self.parent.world_mat_to_local(m)

    @property
    def mask_bounds(self):
        return self.parent.mask_bounds

    @property
    def shape(self):
        return self.parent.shape

    def get_subvolume(self, bounds):
        # assumes bounds given are in local coordinates
        parent_start = self.local_to_parent(bounds.start) if bounds.start is not None else None
        parent_stop = self.local_to_parent(bounds.stop) if bounds.stop is not None else None
        parent_seed = self.local_to_parent(bounds.seed) if bounds.seed is not None else None
        parent_bounds = SubvolumeBounds(start=parent_start,
                                        stop=parent_stop,
                                        seed=parent_seed,
                                        label_id=bounds.label_id,
                                        label_margin=bounds.label_margin)
        return self.parent.get_subvolume(parent_bounds)


class PartitionedVolume(VolumeView):
    """Wrap an existing volume for partitioned access.

    Subvolume accesses to this volume will be offset and clipped to a partition
    of the wrapped volume.

    Parameters
    ----------
    parent : Volume
        The volume to wrap.
    partitioning : iterable of int
        Number of partitions along each axis. Only one axis should be greater
        than 1.
    partition_index : iterable of int
        Index of the partition which this volume will represent.
    """
    def __init__(self, parent, partitioning, partition_index):
        super(PartitionedVolume, self).__init__(
                parent,
                parent.resolution,
                image_data=parent.image_data,
                label_data=parent.label_data,
                mask_data=parent.mask_data)
        self.partitioning = np.asarray(partitioning)
        self.partition_index = np.asarray(partition_index)
        partition_shape = np.floor_divide(np.array(self.parent.shape), self.partitioning)
        self.bounds = ((np.multiply(partition_shape, self.partition_index)).astype(np.int64),
                       (np.multiply(partition_shape, self.partition_index + 1)).astype(np.int64))

    def local_to_parent(self, a):
        return a + self.bounds[0]

    def parent_to_local(self, a):
        return a - self.bounds[0]

    @property
    def mask_bounds(self):
        if self.parent.mask_bounds is None:
            return None
        else:
            bound_min = np.maximum(self.parent.mask_bounds[0], self.bounds[0])
            bound_max = np.minimum(self.parent.mask_bounds[1], self.bounds[1])
            return bound_min, bound_max

    @property
    def shape(self):
        return tuple(self.bounds[1] - self.bounds[0])


class DownsampledVolume(VolumeView):
    """Wrap an existing volume for downsampled access.

    Subvolume accesses to this volume will be downsampled, but continue to use
    the wrapped volume and its data at the original resolution.

    Parameters
    ----------
    parent : Volume
        The volume to wrap.
    downsample : iterable of int
        Integral zoom levels to downsample the wrapped volume.
    """
    def __init__(self, parent, downsample):
        self.scale = np.exp2(downsample).astype(np.int64)
        super(DownsampledVolume, self).__init__(
                parent,
                np.multiply(parent.resolution, self.scale),
                image_data=parent.image_data,
                label_data=parent.label_data,
                mask_data=parent.mask_data)

    def local_to_parent(self, a):
        return np.multiply(a, self.scale)

    def parent_to_local(self, a):
        return np.floor_divide(a, self.scale)

    @property
    def shape(self):
        return tuple(np.floor_divide(np.array(self.parent.shape), self.scale))

    def get_subvolume(self, bounds):
        subvol_shape = bounds.stop - bounds.start
        label_shape = subvol_shape - 2 * bounds.label_margin
        parent_bounds = SubvolumeBounds(self.local_to_parent(bounds.start),
                                        self.local_to_parent(bounds.stop),
                                        label_margin=self.local_to_parent(bounds.label_margin))
        subvol = self.parent.get_subvolume(parent_bounds)
        subvol.image = subvol.image.reshape(
                [subvol_shape[0], self.scale[0],
                 subvol_shape[1], self.scale[1],
                 subvol_shape[2], self.scale[2]]).mean(5).mean(3).mean(1)

        if subvol.label_mask is not None:
            # Downsample body mask by considering blocks where the majority
            # of voxels are in the body to be in the body. Alternatives are:
            # - Conjunction (tends to introduce false splits)
            # - Disjunction (tends to overdilate and merge)
            # - Mode label (computationally expensive)
            if CONFIG.volume.label_downsampling == 'conjunction':
                subvol.label_mask = subvol.label_mask.reshape(
                        [label_shape[0], self.scale[0],
                         label_shape[1], self.scale[1],
                         label_shape[2], self.scale[2]]).all(5).all(3).all(1)
            else:
                subvol.label_mask = subvol.label_mask.reshape(
                        [label_shape[0], self.scale[0],
                         label_shape[1], self.scale[1],
                         label_shape[2], self.scale[2]]).mean(5).mean(3).mean(1) > 0.5

        # Note that this is not a coordinate xform to parent in the typical
        # sense, just a rescaling of the coordinate in the subvolume-local
        # coordinates. Hence no similar call in VolumeView.get_subvolume.
        subvol.seed = self.parent_to_local(subvol.seed)

        return subvol


class SparseWrappedVolume(VolumeView):
    """Wrap a existing volume for memory cached block sparse access."""
    def __init__(self, parent, image_leaf_shape=None, label_leaf_shape=None):
        if image_leaf_shape is None:
            image_leaf_shape = list(CONFIG.model.input_fov_shape)
        if label_leaf_shape is None:
            label_leaf_shape = list(CONFIG.model.input_fov_shape)

        image_data = OctreeVolume(image_leaf_shape,
                                  (np.zeros(3), parent.image_data.shape),
                                  parent.image_data.dtype,
                                  populator=self.image_populator)
        label_data = OctreeVolume(label_leaf_shape,
                                  (np.zeros(3), parent.label_data.shape),
                                  parent.label_data.dtype,
                                  populator=self.label_populator)

        super(SparseWrappedVolume, self).__init__(
                parent,
                parent.resolution,
                image_data=image_data,
                label_data=label_data)

    def image_populator(self, bounds):
        return self.parent.image_data[
                bounds[0][0]:bounds[1][0],
                bounds[0][1]:bounds[1][1],
                bounds[0][2]:bounds[1][2]]

    def label_populator(self, bounds):
        return self.parent.label_data[
                bounds[0][0]:bounds[1][0],
                bounds[0][1]:bounds[1][1],
                bounds[0][2]:bounds[1][2]]


class HDF5Volume(Volume):
    """A volume backed by data views to HDF5 file arrays.

    Parameters
    ----------
    orig_file : str
        Filename of the HDF5 file to load.
    image_dataaset : str
        Full dataset path including groups to the raw image data array.
    label_dataset : str
        Full dataset path including groups to the object label data array.
    """
    @staticmethod
    def from_toml(filename):
        from keras.utils.data_utils import get_file

        volumes = {}
        with open(filename, 'rb') as fin:
            datasets = toml.load(fin).get('dataset', [])
            for dataset in datasets:
                hdf5_file = dataset['hdf5_file']
                if dataset.get('use_keras_cache', False):
                    hdf5_file = get_file(hdf5_file, dataset['download_url'], md5_hash=dataset.get('download_md5', None))
                image_dataset = dataset.get('image_dataset', None)
                label_dataset = dataset.get('label_dataset', None)
                mask_dataset = dataset.get('mask_dataset', None)
                mask_bounds = dataset.get('mask_bounds', None)
                resolution = dataset.get('resolution', None)
                hdf5_pathed_file = os.path.join(os.path.dirname(filename), hdf5_file)
                volume = HDF5Volume(hdf5_pathed_file,
                                    image_dataset,
                                    label_dataset,
                                    mask_dataset,
                                    mask_bounds=mask_bounds)
                # If the volume configuration specifies an explicit resolution,
                # override any provided in the HDF5 itself.
                if resolution:
                    logging.info('Overriding resolution for volume "%s"', dataset['name'])
                    volume.resolution = np.array(resolution)
                volumes[dataset['name']] = volume

        return volumes

    @staticmethod
    def write_file(filename, resolution, **kwargs):
        h5file = h5py.File(filename, 'w')
        config = {'hdf5_file': os.path.basename(filename)}
        channels = ['image', 'label', 'mask']
        default_datasets = {
            'image': 'volumes/raw',
            'label': 'volumes/labels/neuron_ids',
            'mask': 'volumes/labels/mask',
        }
        for channel in channels:
            data = kwargs.get('{}_data'.format(channel), None)
            dataset_name = kwargs.get('{}_dataset'.format(channel), default_datasets[channel])
            if data is not None:
                dataset = h5file.create_dataset(dataset_name, data=data, dtype=data.dtype)
                dataset.attrs['resolution'] = resolution
                config['{}_dataset'.format(channel)] = dataset_name

        h5file.close()

        return config

    def __init__(self, orig_file, image_dataset, label_dataset, mask_dataset, mask_bounds=None):
        logging.debug('Loading HDF5 file "{}"'.format(orig_file))
        self.file = h5py.File(orig_file, 'r')
        self.resolution = None
        self._mask_bounds = tuple(map(np.asarray, mask_bounds)) if mask_bounds is not None else None

        if image_dataset is None and label_dataset is None:
            raise ValueError('HDF5 volume must have either an image or label dataset: {}'.format(orig_file))

        if image_dataset is not None:
            self.image_data = self.file[image_dataset]
            if 'resolution' in self.file[image_dataset].attrs:
                self.resolution = np.array(self.file[image_dataset].attrs['resolution'])

        if label_dataset is not None:
            self.label_data = self.file[label_dataset]
            if 'resolution' in self.file[label_dataset].attrs:
                resolution = np.array(self.file[label_dataset].attrs['resolution'])
                if self.resolution is not None and not np.array_equal(self.resolution, resolution):
                    logging.warning('HDF5 image and label dataset resolutions differ in %s: %s, %s',
                                    orig_file, self.resolution, resolution)
                else:
                    self.resolution = resolution
        else:
            self.label_data = None

        if mask_dataset is not None:
            self.mask_data = self.file[mask_dataset]
        else:
            self.mask_data = None

        if image_dataset is None:
            self.image_data = np.full_like(self.label_data, np.NaN, dtype=np.float32)

        if self.resolution is None:
            self.resolution = np.ones(3)

    def to_memory_volume(self):
        data = ['image_data', 'label_data', 'mask_data']
        data = {
                k: self.world_mat_to_local(getattr(self, k)[:])
                for k in data if getattr(self, k) is not None}
        return NdarrayVolume(self.world_coord_to_local(self.resolution), **data)


class ImageStackVolume(Volume):
    """A volume for block sparse access to image pyramids over HTTP.

    Parameters
    ----------
    bounds : iterable of int
        Shape of the stack at zoom level 0 in pixels.
    resolution : iterable of float
        Resolution of the stack at zoom level 0 in nm.
    tile_width, tile_height : int
        Size of tiles in pixels
    format_url : str
        Format string for building tile URLs from tile parameters.
    zoom_level : int, optional
        Zoom level to use for this volume.
    missing_z : iterable of int, optional
        Voxel z-indices where data is not available.
    image_leaf_shape : tuple of int or ndarray, optional
        Shape of image octree leaves in voxels. Defaults to 10 stacked tiles.
    label_leaf_shape : tuple of int or ndarray, optional
        Shape of label octree leaves in voxels. Defaults to FFN model FOV.
    """
    @staticmethod
    def from_catmaid_stack(stack_info, tile_source_parameters):
        # See https://catmaid.readthedocs.io/en/stable/tile_sources.html
        format_url = {
            1: '{source_base_url}{{z}}/{{row}}_{{col}}_{{zoom_level}}.{file_extension}',
            4: '{source_base_url}{{z}}/{{zoom_level}}/{{row}}_{{col}}.{file_extension}',
            5: '{source_base_url}{{zoom_level}}/{{z}}/{{row}}/{{col}}.{file_extension}',
            7: '{source_base_url}largeDataTileSource/{tile_width}/{tile_height}/'
               '{{zoom_level}}/{{z}}/{{row}}/{{col}}.{file_extension}',
            9: '{source_base_url}{{z}}/{{row}}_{{col}}_{{zoom_level}}.{file_extension}',
        }[tile_source_parameters['tile_source_type']].format(**tile_source_parameters)
        bounds = np.flipud(np.array(stack_info['bounds'], dtype=np.int64))
        resolution = np.flipud(np.array(stack_info['resolution']))
        tile_width = int(tile_source_parameters['tile_width'])
        tile_height = int(tile_source_parameters['tile_height'])
        return ImageStackVolume(bounds, resolution, tile_width, tile_height, format_url,
                                missing_z=stack_info['broken_slices'])

    def __init__(self, bounds, orig_resolution, translation, tile_width, tile_height, 
                 tile_format_url, zoom_level=0, missing_z=None, image_leaf_shape=None):
        self.orig_bounds = bounds
        self.orig_resolution = orig_resolution
        self.translation = translation
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tile_format_url = tile_format_url

        self.zoom_level = int(zoom_level)
        if missing_z is None:
            missing_z = []
        self.missing_z = frozenset(missing_z)
        if image_leaf_shape is None:
            image_leaf_shape = [10, tile_height, tile_width]

        self.scale = np.exp2(np.array([0, self.zoom_level, self.zoom_level])).astype(np.int64)

        data_shape = (np.zeros(3), np.divide(bounds, self.scale).astype(np.int64))
        self.image_data = OctreeVolume(image_leaf_shape,
                                       data_shape,
                                       'float32',
                                       populator=self.image_populator)

        self.label_data = None

    def local_coord_to_world(self, a):
        return self.pixel_coord_to_real(np.matmul(a, self.scale))

    def world_coord_to_local(self, a):
        return np.floor_divide(self.real_coord_to_pixel(a), self.scale)

    def real_coord_to_pixel(self, a):
        return np.floor_divide(a - self.translation, self.orig_resolution)

    def pixel_coord_to_real(self, a):
        return np.matmul(a, self.orig_resolution) + self.translation

    @property
    def resolution(self):
        return self.orig_resolution * np.exp2([0, self.zoom_level, self.zoom_level])

    def downsample(self, resolution):
        downsample = self._get_downsample_from_resolution(resolution)
        zoom_level = np.min(downsample[[self.DIM.X, self.DIM.Y]])
        if zoom_level > 0:
            return ImageStackVolume(
                    self.orig_bounds,
                    self.orig_resolution,
                    self.tile_width,
                    self.tile_height,
                    self.tile_format_url,
                    zoom_level=self.zoom_level + zoom_level,
                    missing_z=self.missing_z,
                    image_leaf_shape=self.image_data.leaf_shape).downsample(resolution)
        if np.all(np.equal(downsample, 0)):
            return self
        return DownsampledVolume(self, downsample)

    def subvolume_bounds_generator(self, sparse_margin=None, **kwargs):
        if sparse_margin is not None:
            if kwargs:
                raise ValueError('sparse_margin can not be combined with other arguments.')
            return self.SparseSubvolumeBoundsGenerator(self, sparse_margin)
        return super(ImageStackVolume, self).subvolume_bounds_generator(**kwargs)

    def get_subvolume(self, bounds):
        if bounds.start is None or bounds.stop is None:
            image_subvol = self.image_data
            label_subvol = self.label_data
        else:
            image_subvol = self.image_data[
                    bounds.start[0]:bounds.stop[0],
                    bounds.start[1]:bounds.stop[1],
                    bounds.start[2]:bounds.stop[2]]
            label_subvol = None

        if np.issubdtype(image_subvol.dtype, np.integer):
            raise ValueError('Sparse volume access does not support image data coercion.')

        seed = bounds.seed
        if seed is None:
            seed = np.array(image_subvol.shape, dtype=np.int64) // 2

        return Subvolume(image_subvol, label_subvol, seed, bounds.label_id)

    def image_populator(self, bounds):
        image_subvol = np.zeros(tuple(bounds[1] - bounds[0]), dtype=np.float32)
        col_range = map(int, (math.floor(bounds[0][self.DIM.X]/self.tile_width),
                              math.ceil(bounds[1][self.DIM.X]/self.tile_width)))
        row_range = map(int, (math.floor(bounds[0][self.DIM.Y]/self.tile_height),
                              math.ceil(bounds[1][self.DIM.Y]/self.tile_height)))
        tile_size = np.array([1, self.tile_height, self.tile_width]).astype(np.int64)
        for z in xrange(bounds[0][self.DIM.Z], bounds[1][self.DIM.Z]):
            if z in self.missing_z:
                image_subvol[int(z - bounds[0][self.DIM.Z]), :, :] = 0
                continue
            for r in xrange(*row_range):
                for c in xrange(*col_range):
                    url = self.tile_format_url.format(zoom_level=self.zoom_level, z=z, row=r, col=c)
                    try:
                        im = np.array(Image.open(requests.get(url, stream=True).raw))
                        # If the image is multichannel, throw our hands up and
                        # just use the first channel.
                        if im.ndim > 2:
                            im = im[:, :, 0].squeeze()
                        im = im / 256.0
                    except IOError:
                        logging.debug('Failed to load tile: %s', url)
                        im = np.full((self.tile_height, self.tile_width), 0, dtype=np.float32)
                    tile_coord = np.array([z, r, c]).astype(np.int64)
                    tile_loc = np.multiply(tile_coord, tile_size)

                    subvol = (np.maximum(np.zeros(3), tile_loc - bounds[0]).astype(np.int64),
                              np.minimum(np.array(image_subvol.shape),
                                         tile_loc + tile_size - bounds[0]).astype(np.int64))
                    tile_sub = (np.maximum(np.zeros(3), bounds[0] - tile_loc).astype(np.int64),
                                np.minimum(tile_size, bounds[1] - tile_loc).astype(np.int64))

                    image_subvol[subvol[0][self.DIM.Z],
                                 subvol[0][self.DIM.Y]:subvol[1][self.DIM.Y],
                                 subvol[0][self.DIM.X]:subvol[1][self.DIM.X]] = \
                        im[tile_sub[0][self.DIM.Y]:tile_sub[1][self.DIM.Y],
                           tile_sub[0][self.DIM.X]:tile_sub[1][self.DIM.X]]

        return image_subvol

    class SparseSubvolumeBoundsGenerator(six.Iterator):
        def __init__(self, volume, margin):
            self.volume = volume
            self.margin = np.asarray(margin).astype(np.int64)
            self.ctr_min = self.margin
            self.ctr_max = (np.array(self.volume.shape) - self.margin - 1).astype(np.int64)
            self.random = np.random.RandomState(CONFIG.random_seed)

        @property
        def shape(self):
            return self.volume.shape

        def __iter__(self):
            return self

        def reset(self):
            self.random.seed(0)

        def __next__(self):
            ctr = np.array([self.random.randint(self.ctr_min[n], self.ctr_max[n])
                            for n in range(3)]).astype(np.int64)
            return SubvolumeBounds(seed=ctr)
