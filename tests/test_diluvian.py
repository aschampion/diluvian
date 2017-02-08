#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_diluvian
----------------------------------

Tests for `diluvian` module.
"""


import numpy as np
import pytest

from diluvian import octrees


def test_octree_bounds():
    clip_bounds = (np.zeros(3), np.array([11, 6, 5]))
    ot = octrees.OctreeMatrix([5, 5, 5], clip_bounds, 'uint8')
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
