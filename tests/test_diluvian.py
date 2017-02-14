#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_diluvian
----------------------------------

Tests for `diluvian` module.
"""


import numpy as np

from diluvian import octrees
from diluvian import regions
from diluvian.config import CONFIG


def test_octree_bounds():
    clip_bounds = (np.zeros(3), np.array([11, 6, 5]))
    ot = octrees.OctreeVolume([5, 5, 5], clip_bounds, 'uint8')
    ot[clip_bounds[0][0]:clip_bounds[1][0],
       clip_bounds[0][1]:clip_bounds[1][1],
       clip_bounds[0][2]:clip_bounds[1][2]] = 6
    assert isinstance(ot.root_node, octrees.UniformNode), "Constant assignment should make root uniform."

    ot[8, 5, 4] = 5
    expected_mat = np.array([[[6], [6]], [[6], [5]]], dtype='uint8')
    assert np.array_equal(ot[7:9, 4:6, 4], expected_mat), "Assignment should break uniformity."

    expected_types = [[[octrees.BranchNode, None], [None, None]],
                      [[octrees.UniformBranchNode, None], [None, None]]]
    for i, col in enumerate(expected_types):
        for j, row in enumerate(col):
            for k, expected_type in enumerate(row):
                if expected_type is None:
                    assert ot.root_node.children[i][j][k] is None, "Clip bounds should make most nodes empty."
                else:
                    assert isinstance(ot.root_node.children[i][j][k], expected_type), "Nodes are wrong type."

    np.testing.assert_almost_equal(ot.fullness(), 2.0/3.0, err_msg='Octree fullness should be relative to clip bounds.')

    ot[10, 5, 4] = 5  # Break the remaining top-level uniform branch node.
    np.testing.assert_almost_equal(ot.fullness(), 1.0, err_msg='Octree fullness should be relative to clip bounds.')


def test_region_moves():
    mock_image = np.zeros(tuple(CONFIG.model.training_fov), dtype='float32')
    region = regions.DenseRegion(mock_image)
    mock_mask = np.zeros(tuple(CONFIG.model.block_size), dtype='float32')
    ctr = np.array(mock_mask.shape) / 2 + 1
    expected_moves = {}
    for i, move in enumerate(map(np.array, [(1, 0, 0), (-1, 0, 0),
                                            (0, 1, 0), (0, -1, 0),
                                            (0, 0, 1), (0, 0, -1)])):
        val = 0.1 * (i + 1)
        coord = ctr + (region.MOVE_DELTA * move) + np.array([2, 2, 2]) * (np.ones(3) - np.abs(move))
        mock_mask[tuple(coord.astype('int64'))] = val
        expected_moves[tuple(move)] = val

    moves = region.get_moves(mock_mask)
    for move in moves:
        np.testing.assert_allclose(expected_moves[tuple(move['move'])], move['v'])
