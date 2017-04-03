# -*- coding: utf-8 -*-
"""Data structures for block sparse labeled components."""


import networkx as nx
import numpy as np
from scipy import ndimage

from .octrees import (
        LeafNode
        )


class SparseLabelVolume(object):
    """Octree-backed blockwise connected component labeled 3D array.

    Parameters
    ----------
    binary_vol : OctreeVolume
        Binary volume to be labeled.
    """
    def __init__(self, binary_vol):
        def empty_uniform(*args):
            return 0

        def label_leaf(mask, *args):
            label_im, _ = ndimage.label(mask)
            return label_im

        self.label_vol = binary_vol.map_copy(np.int32, label_leaf, empty_uniform)
        self.label_graph = nx.Graph()
        self.num_labels = 0

    def make_label_graph(self):
        self.label_graph = nx.Graph()

        for leaf in self.label_vol.iter_leaves():
            leaf_id = tuple(leaf.bounds[0])
            labels = set(np.unique(leaf.data))
            labels.discard(0)
            self.label_graph.add_nodes_from(leaf_id + (l,) for l in labels)

            bounds_slice = [
                slice(leaf.bounds[0][0], leaf.bounds[1][0]),
                slice(leaf.bounds[0][1], leaf.bounds[1][1]),
                slice(leaf.bounds[0][2], leaf.bounds[1][2])]

            # Find connected labels in half of 6-neighboring leaves.
            for axis in range(3):
                u = np.zeros(3)
                u[axis] = 1

                neighbor_id = leaf.bounds[0].copy()
                neighbor_id[axis] = leaf.bounds[1][axis]
                neighbor_id = tuple(neighbor_id)

                # Check if neighbor is outside volume bounds.
                if np.any(np.greater_equal(neighbor_id, self.label_vol.bounds[1])):
                    continue

                bounds = list(bounds_slice)
                bounds[axis] = slice(leaf.bounds[1][axis] - 1, leaf.bounds[1][axis] + 1)

                connected = set()
                interface = self.label_vol[bounds]
                interface = np.rollaxis(interface, axis).T.reshape((-1, 2))
                for l, nl in interface:
                    if l == 0 or nl == 0:
                        continue
                    connected.add((l, nl))

                edges = [(leaf_id + (c[0],), neighbor_id + (c[1],)) for c in connected]
                self.label_graph.add_edges_from(edges)

        self.num_labels = nx.number_connected_components(self.label_graph)

    def get_containing_component(self, pos):
        leaf = self.label_vol.get_node_at(pos)
        if not isinstance(leaf, LeafNode):
            raise ValueError('Position is not in a leaf node.')
        leaf_id = tuple(leaf.bounds[0])
        leaf_pos = pos - leaf.bounds[0]
        leaf_label = leaf.data[leaf_pos[0], leaf_pos[1], leaf_pos[2]]

        if leaf_label == 0:
            raise ValueError('Position is not in a connected component.')

        label_id = leaf_id + (leaf_label,)
        return nx.node_connected_component(self.label_graph, label_id)

    def get_component_volume(self, component):
        def empty_uniform(*args):
            return 0

        def delabel_leaf(mask, leaf):
            leaf_id = tuple(leaf.bounds[0])
            leaf_labels = np.array([l[3] for l in component if l[0:3] == leaf_id], dtype=mask.dtype)

            return np.in1d(mask, leaf_labels).reshape(mask.shape).astype(np.int32)

        return self.label_vol.map_copy(np.int32, delabel_leaf, empty_uniform)
