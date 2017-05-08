# -*- coding: utf-8 -*-


import collections
import csv
import itertools
import webbrowser

import neuroglancer
import numpy as np


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
        print self

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


# Taken from the python docs itertools recipes
def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))
