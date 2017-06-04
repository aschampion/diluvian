# -*- coding: utf-8 -*-


from __future__ import print_function

import collections
import csv
import itertools
import webbrowser

import neuroglancer
import numpy as np
import six
from six.moves import input as raw_input


class WrappedViewer(neuroglancer.Viewer):
    def __init__(self, voxel_coordinates=None, **kwargs):
        super(WrappedViewer, self).__init__(**kwargs)
        self.voxel_coordinates = voxel_coordinates

    def get_json_state(self):
        state = super(WrappedViewer, self).get_json_state()
        if self.voxel_coordinates is not None:
            if 'navigation' not in state:
                state['navigation'] = collections.OrderedDict()
            if 'pose' not in state['navigation']:
                state['navigation']['pose'] = collections.OrderedDict()
            if 'position' not in state['navigation']['pose']:
                state['navigation']['pose']['position'] = collections.OrderedDict()
            state['navigation']['pose']['position']['voxelCoordinates'] = map(int, list(self.voxel_coordinates))
        return state

    def open_in_browser(self):
        webbrowser.open_new_tab(str(self))

    def print_view_prompt(self):
        print(self)

        while True:
            s = raw_input('Press v, enter to open in browser, or enter to close...')
            if s == 'v':
                self.open_in_browser()
            else:
                break


def extend_keras_history(a, b):
    a.epoch.extend(b.epoch)
    for k, v in b.history.items():
        a.history.setdefault(k, []).extend(v)


def write_keras_history_to_csv(history, filename):
    """Write Keras history to a CSV file.

    If the file already exists it will be overwritten.

    Parameters
    ----------
    history : keras.callbacks.History
    filename : str
    """
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        metric_cols = history.history.keys()
        indices = [i[0] for i in sorted(enumerate(metric_cols), key=lambda x: x[1])]
        metric_cols.sort()
        cols = ['epoch'] + metric_cols
        sorted_metrics = history.history.values()
        sorted_metrics = [sorted_metrics[i] for i in indices]
        writer.writerow(cols)
        for row in zip(history.epoch, *sorted_metrics):
            writer.writerow(row)


def get_color_shader(channel):
    value_str = 'toNormalized(getDataValue(0))'
    channels = ['0', '0', '0', value_str]
    channels[channel] = '1'
    shader = """
void main() {{
  emitRGBA(vec4({}));
}}
""".format(', '.join(channels))
    return shader


def pad_dims(x):
    """Add single-dimensions to the beginning and end of an array."""
    return np.expand_dims(np.expand_dims(x, x.ndim), 0)


def get_nonzero_aabb(a):
    """Get the axis-aligned bounding box of nonzero elements of a 3D array.

    Parameters
    ----------
    a : ndarray
        A 3D NumPpy array.

    Returns
    -------
    tuple of ndarray
    """
    mask_min = []
    mask_max = []

    for axes in [(1, 2), (0, 2), (0, 1)]:
        proj = np.any(a, axis=axes)
        w = np.where(proj)[0]
        if w.size:
            amin, amax = w[[0, -1]]
        else:
            amin, amax = 0, 0

        mask_min.append(amin)
        mask_max.append(amax)

    mask_min = np.array(mask_min, dtype=np.int64)
    mask_max = np.array(mask_max, dtype=np.int64)

    return mask_min, mask_max


class Roundrobin(six.Iterator):
    """Iterate over a collection of iterables, pulling one item from each in
    a cycle.

    Based on a generator function recipe credited to George Sakkis on the
    python docs itertools recipes.

    Examples
    --------
    >>> list(Roundrobin('ABC', 'D', 'EF'))
    ['A', 'D', 'E', 'B', 'F', 'C']
    """

    def __init__(self, *iterables):
        self.iterables = iterables
        self.pending = len(self.iterables)
        self.nexts = itertools.cycle(self.iterables)

    def __iter__(self):
        return self

    def reset(self):
        for it in self.iterables:
            iter(it).reset()
        self.pending = len(self.iterables)
        self.nexts = itertools.cycle(self.iterables)

    def __next__(self):
        while self.pending:
            try:
                for nextgen in self.nexts:
                    return six.next(nextgen)
            except StopIteration:
                self.pending -= 1
                self.nexts = itertools.cycle(itertools.islice(self.nexts, self.pending))
        raise StopIteration()
