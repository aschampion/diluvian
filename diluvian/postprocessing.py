# -*- coding: utf-8 -*-
"""Segmentation processing and skeletonization after flood filling."""


from __future__ import division
from __future__ import print_function

import csv
import logging

import numpy as np
from scipy import ndimage

from .config import CONFIG
from .octrees import OctreeVolume
from .util import get_nonzero_aabb


class Body(object):
    def __init__(self, mask, seed):
        self.mask = mask
        self.seed = seed

    def _get_bounded_mask(self, closing_shape=None):
        if isinstance(self.mask, OctreeVolume):
            # If this is a sparse volume, materialize it to memory.
            bounds = self.mask.get_leaf_bounds()
            mask = self.mask[map(slice, bounds[0], bounds[1])]
            # Crop the mask and bounds to nonzero region of the mask.
            mask_min, mask_max = get_nonzero_aabb(mask)
            bounds[0] += mask_min
            bounds[1] -= np.array(mask.shape) - mask_max
            mask = mask[map(slice, mask_min, mask_max)]
            assert mask.shape == tuple(bounds[1] - bounds[0]), \
                'Bounds shape ({}) and mask shape ({}) differ.'.format(bounds[1] - bounds[0], mask.shape)
        else:
            bounds = (np.zeros(3), np.array(self.mask.shape))
            mask = self.mask

        if closing_shape is not None:
            mask = ndimage.grey_closing(mask, size=closing_shape, mode='nearest')

        return mask, bounds

    def get_largest_component(self, closing_shape=None):
        mask, bounds = self._get_bounded_mask(closing_shape)

        label_im, num_labels = ndimage.label(mask)
        label_sizes = ndimage.sum(mask, label_im, range(num_labels + 1))
        label_im[(label_sizes < label_sizes.max())[label_im]] = 0
        label_im = np.minimum(label_im, 1)

        if label_im[tuple(self.seed - bounds[0])] == 0:
            logging.warning('Seed voxel ({}) is not in connected component.'.format(np.array_str(self.seed)))

        return label_im, bounds

    def get_seeded_component(self, closing_shape=None):
        mask, bounds = self._get_bounded_mask(closing_shape)

        label_im, _ = ndimage.label(mask)
        seed_label = label_im[tuple(self.seed - bounds[0])]
        if seed_label == 0:
            raise ValueError('Seed voxel (%s) is not in body.', np.array_str(self.seed))
        label_im[label_im != seed_label] = 0
        label_im[label_im == seed_label] = 1

        return label_im, bounds

    def to_swc(self, filename):
        component, bounds = self.get_largest_component(closing_shape=CONFIG.postprocessing.closing_shape)
        print('Skeleton is within {}, {}'.format(np.array_str(bounds[0]), np.array_str(bounds[1])))
        skel = skeletonize_component(component)
        swc = skeleton_to_swc(skel, bounds[0], CONFIG.volume.resolution)
        with open(filename, 'w') as swcfile:
            writer = csv.writer(swcfile, delimiter=' ', quoting=csv.QUOTE_NONE)
            writer.writerows(swc)


def skeletonize_component(component):
    import skeletopyze

    params = skeletopyze.Parameters()
    res = skeletopyze.point_f3()
    for i in range(3):
        res[i] = CONFIG.volume.resolution[i]

    print('Skeletonizing...')
    skel = skeletopyze.get_skeleton_graph(component.astype(np.int32), params, res)

    return skel


def skeleton_to_swc(skeleton, offset, resolution):
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(skeleton.nodes())
    g.add_edges_from((e.u, e.v) for e in skeleton.edges())

    # Find a directed tree for mapping to a skeleton.
    if nx.number_of_nodes(g) > 1:
        # This discards cyclic edges in the graph.
        t = nx.bfs_tree(nx.minimum_spanning_tree(g), g.nodes()[0])
    else:
        t = nx.DiGraph()
        t.add_nodes_from(g)
    # Copy node attributes
    for n in t.nodes_iter():
        loc = skeleton.locations(n)
        # skeletopyze is z, y, x (as it should be).
        loc = np.array(loc)
        loc = np.multiply(loc + offset, resolution)
        t.node[n].update({'x': loc[0],
                          'y': loc[1],
                          'z': loc[2],
                          'radius': skeleton.diameters(n) / 2.0})

    # Set parent node ID
    for n, nbrs in t.adjacency_iter():
        for nbr in nbrs:
            t.node[nbr]['parent_id'] = n
            if 'radius' not in t.node[nbr]:
                t.node[nbr]['radius'] = -1

    return [[
        node_id,
        0,
        n['x'], n['y'], n['z'],
        n['radius'],
        n.get('parent_id', -1)] for node_id, n in t.nodes(data=True)]
