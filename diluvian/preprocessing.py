# -*- coding: utf-8 -*-
"""Volume preprocessing for seed generation and data augmentation."""


from __future__ import division

import logging

import numpy as np
from scipy import ndimage

from .config import CONFIG
from .util import WrappedViewer


def intensity_distance_seeds(image_data, visualize=False):
    """Create seed locations maximally distant from a Sobel filter.

    Parameters
    ----------
    image_data : ndarray

    Returns
    -------
    list of ndarray
    """
    # Late import as this is the only function using Scikit.
    from skimage.morphology import extrema

    logging.debug('Running Sobel filter on image shape %s', image_data.shape)
    sobel = ndimage.generic_gradient_magnitude(image_data, ndimage.prewitt)
    logging.debug('Running distance transform on image shape %s', image_data.shape)

    # For low res images the sobel histogram is unimodal. For now just
    # threshold the histogram at the mean.
    thresh = sobel < np.mean(sobel)
    transform = ndimage.distance_transform_cdt(thresh)
    # Remove missing sections from distance transform.
    transform[image_data == 0] = 0
    logging.debug('Finding local maxima of image shape %s', image_data.shape)
    skmax = extrema.local_maxima(transform)

    if visualize:
        viewer = WrappedViewer()
        viewer.add(image_data, name='Image')
        viewer.add(sobel, name='Filtered')
        viewer.add(thresh.astype(np.float), name='Thresholded')
        viewer.add(transform.astype(np.float), name='Distance')
        viewer.add(skmax, name='Seeds')
        viewer.print_view_prompt()

    seeds = np.transpose(np.nonzero(skmax))

    return seeds


def grid_seeds(image_data):
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
