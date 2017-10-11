#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import glob
import os
import re

import argparse
import h5py
import numpy as np
import pytoml as toml

from diluvian.util import get_nonzero_aabb


def create_dataset_conf_from_files(path, file_pattern, name_regex, name_format, mask_bounds=True):
    pathspec = path + file_pattern
    name_regex = re.compile(name_regex)

    datasets = []

    for pathname in glob.iglob(pathspec):
        filename = os.path.basename(pathname)
        name = name_format.format(*name_regex.match(filename).groups())
        ds = {
                'name': name,
                'hdf5_file': pathname,
                'image_dataset': 'volumes/raw',
                'label_dataset': 'volumes/labels/neuron_ids',
                'mask_dataset': 'volumes/labels/mask',
                'resolution': [40, 4, 4],
        }

        if mask_bounds:
            print('Finding mask bounds for {}'.format(filename))
            f = h5py.File(pathname, 'r')
            d = f[ds['mask_dataset']]
            mask_data = d[:]
            mask_min, mask_max = get_nonzero_aabb(mask_data)

            ds['mask_bounds'] = [mask_min, mask_max]
            f.close()

        datasets.append(ds)

    return {'dataset': datasets}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dataset TOML from a directory of HDF5 files.')

    parser.add_argument(
        '--file-pattern', dest='file_pattern', default='sample_[ABC]*hdf',
        help='Glob for HDF5 volume filenames.')
    parser.add_argument(
        '--name-regex', dest='name_regex', default=r'sample_([ABC])(.*).hdf',
        help='Regex for extracting volume name from filenames.')
    parser.add_argument(
        '--name-format', dest='name_format', default='Sample {} ({})',
        help='Format string for creating volume names from name regex matches.')
    parser.add_argument(
        'path', default=None,
        help='Path to the HDF5 volume files.')
    parser.add_argument(
        'dataset_file', default=None,
        help='Name for the TOML dataset file that will be created.')

    args = parser.parse_args()

    conf = create_dataset_conf_from_files(args.path, args.file_pattern, args.name_regex, args.name_format)
    print('Found {} datasets.'.format(len(conf['dataset'])))

    with open(args.dataset_file, 'wb') as tomlfile:
        tomlfile.write(toml.dumps(conf))
