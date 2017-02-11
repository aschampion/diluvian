# -*- coding: utf-8 -*-
"""Global configuration objects.

This module contains boilerplate configuration objects for storing and loading
configuration state.
"""


import os

import numpy as np
import pytoml as toml


class BaseConfig:
    def __str__(self):
        sanitized = {}
        for k, v in self.__dict__.iteritems():
            if isinstance(v, np.ndarray):
                sanitized[k] = v.tolist()
            else:
                sanitized[k] = v
        return toml.dumps(sanitized)

    __repr__ = __str__


class VolumeConfig(BaseConfig):
    def __init__(self, settings):
        self.downsample = np.array(settings.get('downsample', [0, 0, 0]))
        self.resolution = np.array(settings.get('resolution', [1, 1, 1]))


class ModelConfig(BaseConfig):
    def __init__(self, settings):
        self.block_size = np.array(settings.get('block_size', [33, 33, 17]))
        self.v_true = float(settings.get('v_true', 0.095))
        self.v_false = float(settings.get('v_false', 0.05))
        self.t_move = float(settings.get('t_move', 0.9))
        self.training_fov = np.array(settings.get('training_fov', self.block_size + ((self.block_size - 1) / 2)))


class NetworkConfig(BaseConfig):
    def __init__(self, settings):
        self.factory = str(settings.get('factory'))
        self.num_modules = int(settings.get('num_modules', 8))
        self.convolution_dim = np.array(settings.get('convolution_dim', [3, 3, 3]))
        self.convolution_filters = int(settings.get('convolution_filters', 32))
        self.output_activation = str(settings.get('output_activation', 'sigmoid'))


class OptimizerConfig(BaseConfig):
    def __init__(self, settings):
        self.klass = str(settings.get('class', 'SGD'))
        self.kwargs = {k: v for k, v in settings.iteritems() if k != 'class'}


class TrainingConfig(BaseConfig):
    def __init__(self, settings):
        self.num_gpus = int(settings.get('num_gpus', 1))
        self.gpu_batch_size = int(settings.get('gpu_batch_size', 8))
        self.batch_size = self.num_gpus * self.gpu_batch_size
        self.training_size = int(settings.get('training_size', 256))
        self.validation_size = int(settings.get('validation_size', 256))
        self.simple_train_epochs = int(settings.get('simple_train_epochs', 10))
        self.total_epochs = int(settings.get('total_epochs', 100))
        self.fill_factor_bins = settings.get('fill_factor_bins', None)
        if self.fill_factor_bins is not None:
            self.fill_factor_bins = np.array(self.fill_factor_bins)
        self.partitions = np.array(settings.get('partitions', [0, 0, 0]))
        self.training_partition = np.array(settings.get('training_partition', [0, 0, 0]))
        self.validation_partition = np.array(settings.get('validation_partition', [0, 0, 1]))
        self.patience = int(np.array(settings.get('patience', 10)))


class Config(BaseConfig):
    def from_toml(self, *filenames):
        settings = []
        for filename in filenames:
            with open(filename, 'rb') as fin:
                settings.append(toml.load(fin))

        return self.__init__(settings)

    def __init__(self, settings_collection=None):
        if settings_collection is not None:
            settings = settings_collection[0].copy()
            for s in settings_collection:
                for c in s:
                    if c in settings:
                        settings[c].update(s[c])
                    else:
                        settings[c] = s[c]
        else:
            settings = {}

        self.volume = VolumeConfig(settings.get('volume', {}))
        self.model = ModelConfig(settings.get('model', {}))
        self.network = NetworkConfig(settings.get('network', {}))
        self.optimizer = OptimizerConfig(settings.get('optimizer', {}))
        self.training = TrainingConfig(settings.get('training', {}))


CONFIG = Config()
CONFIG.from_toml(os.path.join(os.path.dirname(__file__), '..', 'conf', 'default.toml'))
