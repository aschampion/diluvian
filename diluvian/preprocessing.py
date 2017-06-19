# -*- coding: utf-8 -*-
"""Volume preprocessing for seed generation and data augmentation."""


from __future__ import division

import logging

import numpy as np
from scipy import ndimage
from six.moves import range as xrange

from .config import CONFIG
from .util import (
        get_color_shader,
        WrappedViewer,
)


def intensity_distance_seeds(image_data, resolution, axis=0, erosion_radius=12, min_sep=12, visualize=False):
    """Create seed locations maximally distant from a Sobel filter.

    Parameters
    ----------
    image_data : ndarray
    resolution : ndarray
    axis : int, optional
        Axis along which to slices volume to generate seeds in 2D. If
        None volume is processed in 3D.
    erosion_radius : int, optional
        L_infinity norm radius of the structuring element for eroding
        components.
    min_sep : int, optional
        L_infinity minimum separation of seeds in nanometers.

    Returns
    -------
    list of ndarray
    """
    # Late import as this is the only function using Scikit.
    from skimage import morphology

    structure = np.ones(np.floor_divide(erosion_radius, resolution) * 2 + 1)

    if axis is None:
        def slices():
            yield [slice(None), slice(None), slice(None)]
    else:
        structure = structure[axis]

        def slices():
            for i in xrange(image_data.shape[axis]):
                s = map(slice, [None] * 3)
                s[axis] = i
                yield s

    sobel = np.zeros_like(image_data)
    thresh = np.zeros_like(image_data)
    transform = np.zeros_like(image_data)
    skmax = np.zeros_like(image_data)
    for s in slices():
        image_slice = image_data[s]
        if axis is not None and not np.any(image_slice):
            logging.debug('Skipping blank slice.')
            continue
        logging.debug('Running Sobel filter on image shape %s', image_data.shape)
        sobel[s] = ndimage.generic_gradient_magnitude(image_slice, ndimage.prewitt)
        # sobel = ndimage.grey_dilation(sobel, size=(5,5,3))
        logging.debug('Running distance transform on image shape %s', image_data.shape)

        # For low res images the sobel histogram is unimodal. For now just
        # threshold the histogram at the mean.
        thresh[s] = sobel[s] < np.mean(sobel[s])
        thresh[s] = ndimage.binary_erosion(thresh[s], structure=structure)
        transform[s] = ndimage.distance_transform_cdt(thresh[s])
        # Remove missing sections from distance transform.
        transform[s][image_slice == 0] = 0
        logging.debug('Finding local maxima of image shape %s', image_data.shape)
        skmax[s] = morphology.thin(morphology.extrema.local_maxima(transform[s]))

    if visualize:
        viewer = WrappedViewer()
        viewer.add(image_data, name='Image')
        viewer.add(sobel, name='Filtered')
        viewer.add(thresh.astype(np.float), name='Thresholded')
        viewer.add(transform.astype(np.float), name='Distance')
        viewer.add(skmax, name='Seeds', shader=get_color_shader(0, normalized=False))
        viewer.print_view_prompt()

    mask = np.zeros(np.floor_divide(erosion_radius, resolution) + 1)
    mask[0, 0, 0] = 1
    seeds = np.transpose(np.nonzero(skmax))
    for seed in seeds:
        if skmax[tuple(seed)]:
            lim = np.minimum(mask.shape, skmax.shape - seed)
            skmax[map(slice, seed, seed + lim)] = mask[map(slice, lim)]

    seeds = np.transpose(np.nonzero(skmax))

    return seeds


def grid_seeds(image_data, _):
    """Create seed locations in a volume on a uniform grid.

    Parameters
    ----------
    image_data : ndarray

    Returns
    -------
    list of ndarray
    """
    seeds = []
    shape = image_data.shape
    grid_size = (CONFIG.model.output_fov_shape - 1) // 2
    for x in range(grid_size[0], shape[0], grid_size[0]):
        for y in range(grid_size[1], shape[1], grid_size[1]):
            for z in range(grid_size[2], shape[2], grid_size[2]):
                seeds.append(np.array([x, y, z], dtype=np.int32))

    return seeds


# Note that these must be added separately to the CLI.
SEED_GENERATORS = {
    'grid': grid_seeds,
    'sobel': intensity_distance_seeds,
}
