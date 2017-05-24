# -*- coding: utf-8 -*-
"""Simple octree data structures for block sparse 3D arrays."""


from __future__ import division

import numpy as np


class OctreeVolume(object):
    """Octree-backed block sparse 3D array.

    This is a trivial implementation of an octree with NumPy ndarray leaves for
    block sparse volume access. This allows oblivious in-memory access of
    dense regions spanning out-of-memory volumes by providing read leaves
    via a populator. For writing, the octree supports uniform value terminal
    nodes at every level, so that only non-uniform data must be written to
    leaf level.

    Parameters
    ----------
    leaf_shape : tuple of int or ndarray
        Shape of tree leaves in voxels.
    bounds : tuple of tuple of int or ndarray
        The lower and upper coordinate bounds of the volume, in voxels.
    dtype : numpy.data-type
    populator : function, optional
        A function taking a tuple of ndarray bounds for the coordinates of
        the subvolume to populate and returning the data for that subvolume.
    """

    def __init__(self, leaf_shape, bounds, dtype, populator=None):
        self.leaf_shape = np.asarray(leaf_shape).astype(np.int64)
        self.bounds = (np.asarray(bounds[0], dtype=np.int64),
                       np.asarray(bounds[1], dtype=np.int64))
        self.dtype = np.dtype(dtype)
        self.populator = populator
        ceil_bounds = self.leaf_shape * \
            np.exp2(np.ceil(np.log2((self.bounds[1] - self.bounds[0]) /
                                    self.leaf_shape.astype('float64')))).astype(np.int64).max()
        self.root_node = BranchNode(self, (self.bounds[0], self.bounds[0] + ceil_bounds), clip_bound=self.bounds[1])

    @property
    def shape(self):
        return tuple(self.root_node.get_size())

    def get_checked_np_key(self, key):
        # Special exception for [:] for uniform assignment.
        if isinstance(key, slice) and key.start is None and key.stop is None:
            return self.bounds

        if not hasattr(key, '__len__') or len(key) != 3:
            raise IndexError('Octrees may only be indexed in 3 dimensions')

        # Convert keys to two numpy arrays for ease.
        npkey = (np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64))
        for i, k in enumerate(key):
            if isinstance(k, slice):
                if k.step is not None:
                    raise IndexError('Octrees do not yet support step slicing')
                npkey[0][i] = k.start if k.start is not None else self.bounds[0][i]
                npkey[1][i] = k.stop if k.stop is not None else self.bounds[1][i]
            else:
                npkey[0][i] = k
                npkey[1][i] = k + 1

        if np.any(np.less(npkey[0], self.bounds[0])) or \
           np.any(np.greater(npkey[1], self.bounds[1])) or \
           np.any(np.greater_equal(npkey[0], npkey[1])):
            raise IndexError('Invalid indices: outside bounds or empty interval: '
                             '{} (bounds {})'.format(str(key), str(self.bounds)))

        return npkey

    def __getitem__(self, key):
        npkey = self.get_checked_np_key(key)

        return self.root_node[npkey]

    def __setitem__(self, key, value):
        npkey = self.get_checked_np_key(key)

        self.root_node[npkey] = value

    def iter_leaves(self):
        """Iterator over all non-uniform leaf nodes.

        Yields
        ------
        LeafNode
        """
        for leaf in self.root_node.iter_leaves():
            yield leaf

    def get_leaf_bounds(self):
        bounds = [np.array(self.bounds[1]), np.array(self.bounds[0])]
        for leaf in self.iter_leaves():
            bounds[0] = np.minimum(bounds[0], leaf.bounds[0])
            bounds[1] = np.maximum(bounds[1], leaf.bounds[1])

        bounds[0] = np.maximum(bounds[0], self.bounds[0])
        bounds[1] = np.minimum(bounds[1], self.bounds[1])

        return bounds

    def map_copy(self, dtype, leaf_map, uniform_map):
        """Create a copy of this octree by mapping node data.

        Note that because leaves and uniform nodes can have separate mapping,
        the ranges of this tree and the copied tree may not be bijective.

        Populators are not copied.

        Parameters
        ----------
        dtype : numpy.data-type
            Data type for the constructed copy
        leaf_map : function
            Function mapping leaf node data for the constructed copy.
        uniform_map : function
            Function mapping uniform node values.

        Returns
        -------
        OctreeVolume
            Copied octree with the same structure as this octree.
        """
        copy = OctreeVolume(self.leaf_shape, self.bounds, dtype)
        copy.root_node = self.root_node.map_copy(copy, leaf_map, uniform_map)
        return copy

    def fullness(self):
        potential_leaves = np.prod(np.ceil(np.true_divide(self.bounds[1] - self.bounds[0], self.leaf_shape)))
        return self.root_node.count_leaves() / float(potential_leaves)

    def get_volume(self):
        return self

    def replace_child(self, child, replacement):
        if child != self.root_node:
            raise ValueError('Attempt to replace unknown child')

        self.root_node = replacement


class Node(object):
    def __init__(self, parent, bounds, clip_bound=None):
        self.parent = parent
        self.bounds = (bounds[0].copy(), bounds[1].copy())
        self.clip_bound = clip_bound

    def count_leaves(self):
        return 0

    def get_intersection(self, key):
        return (np.maximum(self.bounds[0], key[0]),
                np.minimum(self.bounds[1], key[1]))

    def get_size(self):
        if self.clip_bound is not None:
            return self.clip_bound - self.bounds[0]
        return self.bounds[1] - self.bounds[0]

    def get_volume(self):
        return self.parent.get_volume()

    def replace(self, replacement):
        self.parent.replace_child(self, replacement)
        self.parent = None


class BranchNode(Node):
    def __init__(self, parent, bounds, **kwargs):
        super(BranchNode, self).__init__(parent, bounds, **kwargs)
        self.midpoint = (self.bounds[1] + self.bounds[0]) // 2
        self.children = [[[None for _ in range(2)] for _ in range(2)] for _ in range(2)]

    def count_leaves(self):
        return sum(c.count_leaves() for s in self.children for r in s for c in r if c is not None)

    def iter_leaves(self):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    child = self.children[i][j][k]
                    if child is None or isinstance(child, UniformNode):
                        continue
                    for leaf in child.iter_leaves():
                        yield leaf

    def map_copy(self, copy_parent, leaf_map, uniform_map):
        copy = BranchNode(copy_parent, self.bounds, clip_bound=self.clip_bound)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    child = self.children[i][j][k]
                    if child is None:
                        copy.children[i][j][k] = None
                    else:
                        copy.children[i][j][k] = child.map_copy(copy, leaf_map, uniform_map)
        return copy

    def get_children_mask(self, key):
        p = (np.less(key[0], self.midpoint),
             np.greater(key[1], self.midpoint))

        # TODO must be some way to do combinatorial ops like this with numpy.
        return np.where([[[p[i][0] and p[j][1] and p[k][2] for k in range(2)] for j in range(2)] for i in range(2)])

    def get_child_bounds(self, i, j, k):
        mins = (self.bounds[0], self.midpoint)
        maxs = (self.midpoint, self.bounds[1])
        child_bounds = (np.array((mins[i][0], mins[j][1], mins[k][2])),
                        np.array((maxs[i][0], maxs[j][1], maxs[k][2])))
        if self.clip_bound is not None:
            clip_bound = np.minimum(child_bounds[1], self.clip_bound)
            if np.array_equal(clip_bound, child_bounds[1]):
                clip_bound = None
        else:
            clip_bound = None

        return (child_bounds, clip_bound)

    def __getitem__(self, key):
        inds = self.get_children_mask(key)

        for i, j, k in zip(*inds):
            if self.children[i][j][k] is None:
                self.populate_child(i, j, k)

        chunk = np.empty(tuple(key[1] - key[0]), self.get_volume().dtype)
        for i, j, k in zip(*inds):
            child = self.children[i][j][k]
            subchunk = child.get_intersection(key)
            ind = (subchunk[0] - key[0], subchunk[1] - key[0])
            chunk[ind[0][0]:ind[1][0],
                  ind[0][1]:ind[1][1],
                  ind[0][2]:ind[1][2]] = child[subchunk]

        return chunk

    def __setitem__(self, key, value):
        if (not hasattr(value, '__len__') or len(value) == 1) and \
           np.array_equal(key[0], self.bounds[0]) and \
           np.array_equal(key[1], self.clip_bound):
            self.replace(UniformBranchNode(self.parent, self.bounds, self.get_volume().dtype, value,
                                           clip_bound=self.clip_bound))
            return

        inds = self.get_children_mask(key)

        for i, j, k in zip(*inds):
            if self.children[i][j][k] is None:
                self.populate_child(i, j, k)

        for i, j, k in zip(*inds):
            child = self.children[i][j][k]
            subchunk = child.get_intersection(key)
            ind = (subchunk[0] - key[0], subchunk[1] - key[0])
            if isinstance(value, np.ndarray):
                child[subchunk] = value[ind[0][0]:ind[1][0],
                                        ind[0][1]:ind[1][1],
                                        ind[0][2]:ind[1][2]]
            else:
                child[subchunk] = value

    def populate_child(self, i, j, k):
        volume = self.get_volume()
        if volume.populator is None:
            raise ValueError('Attempt to retrieve unpopulated region without octree populator')

        child_bounds, child_clip_bound = self.get_child_bounds(i, j, k)
        child_shape = child_bounds[1] - child_bounds[0]
        if np.any(np.less_equal(child_shape, volume.leaf_shape)):
            populator_bounds = [child_bounds[0].copy(), child_bounds[1].copy()]
            if child_clip_bound is not None:
                populator_bounds[1] = np.minimum(populator_bounds[1], child_clip_bound)
            data = volume.populator(populator_bounds).astype(volume.dtype)
            child = LeafNode(self, child_bounds, data)
        else:
            child = BranchNode(self, child_bounds, clip_bound=child_clip_bound)

        self.children[i][j][k] = child

    def replace_child(self, child, replacement):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if child == self.children[i][j][k]:
                        self.children[i][j][k] = replacement
                        return

        raise ValueError('Attempt to replace unknown child')


class LeafNode(Node):
    def __init__(self, parent, bounds, data):
        super(LeafNode, self).__init__(parent, bounds)
        self.data = data.copy()

    def count_leaves(self):
        return 1

    def iter_leaves(self):
        yield self

    def map_copy(self, copy_parent, leaf_map, uniform_map):
        copy = LeafNode(copy_parent, self.bounds, leaf_map(self.data))
        return copy

    def __getitem__(self, key):
        ind = (key[0] - self.bounds[0], key[1] - self.bounds[0])
        return self.data[ind[0][0]:ind[1][0],
                         ind[0][1]:ind[1][1],
                         ind[0][2]:ind[1][2]]

    def __setitem__(self, key, value):
        ind = (key[0] - self.bounds[0], key[1] - self.bounds[0])
        self.data[ind[0][0]:ind[1][0],
                  ind[0][1]:ind[1][1],
                  ind[0][2]:ind[1][2]] = value


class UniformNode(Node):
    def __init__(self, parent, bounds, dtype, value, **kwargs):
        super(UniformNode, self).__init__(parent, bounds, **kwargs)
        self.value = value
        self.dtype = dtype

    def __getitem__(self, key):
        return np.full(tuple(key[1] - key[0]), self.value, dtype=self.dtype)

    def map_copy(self, copy_parent, leaf_map, uniform_map):
        copy = type(self)(copy_parent, self.bounds, copy_parent.get_volume().dtype,
                          uniform_map(self.value), clip_bound=self.clip_bound)
        return copy


class UniformBranchNode(UniformNode):
    def __setitem__(self, key, value):
        replacement = BranchNode(self.parent, self.bounds, clip_bound=self.clip_bound)
        volume = self.get_volume()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    child_bounds, child_clip_bound = replacement.get_child_bounds(i, j, k)
                    # If this child is entirely outside the clip bounds, it will never be accessed
                    # or populated and thus can be omitted.
                    if child_clip_bound is not None and np.any(np.greater_equal(child_bounds[0], child_clip_bound)):
                        continue
                    child_shape = child_bounds[1] - child_bounds[0]
                    if np.any(np.less_equal(child_shape, volume.leaf_shape)):
                        child = UniformLeafNode(replacement, child_bounds, self.dtype, self.value)
                    else:
                        child = UniformBranchNode(replacement, child_bounds, self.dtype, self.value,
                                                  clip_bound=child_clip_bound)
                    replacement.children[i][j][k] = child
        self.replace(replacement)
        replacement[key] = value


class UniformLeafNode(UniformNode):
    def __setitem__(self, key, value):
        replacement = LeafNode(self.parent, self.bounds, self[self.bounds])
        self.replace(replacement)
        replacement[key] = value

    def count_leaves(self):
        return 1
