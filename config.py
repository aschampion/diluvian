# -*- coding: utf-8 -*-
"""Global configuration objects.

This module contains boilerplate configuration objects for storing and loading
configuration state.
"""


import os

import numpy as np
import pytoml as toml


class VolumeConfig(object):
    def __init__(self, settings):
        self.downsample = np.array(settings.get('downsample', [0, 0, 0]))
        self.resolution = np.array(settings.get('resolution', [1, 1, 1]))


class ModelConfig(object):
    def __init__(self, settings):
        self.block_size = np.array(settings.get('block_size', [33, 33, 17]))
        self.v_true = float(settings.get('v_true', 0.095))
        self.v_false = float(settings.get('v_false', 0.05))
        self.t_move = float(settings.get('t_move', 0.9))
        self.training_fov = np.array(settings.get('training_fov', self.block_size + ((self.block_size - 1) / 2)))


class NetworkConfig(object):
    def __init__(self, settings):
        self.num_modules = int(settings.get('num_modules', 8))
        self.convolution_dim = np.array(settings.get('convolution_dim', [3, 3, 3]))


class OptimizerConfig(object):
    def __init__(self, settings):
        self.learning_rate = float(settings.get('learning_rate', 0.001))
        self.momentum = float(settings.get('momentum', 0.0))
        self.nesterov_momentum = bool(settings.get('nesterov_momentum', False))


class TrainingConfig(object):
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


class Config(object):
    @staticmethod
    def from_toml(*filenames):
        settings = []
        for filename in filenames:
            with open(filename, 'rb') as fin:
                settings.append(toml.load(fin))

        return Config(settings)

    def __init__(self, settings_collection=None):
        if settings_collection is not None:
            settings = settings_collection[0].copy()
            for s in settings_collection:
                settings.update(s)
        else:
            settings = {}

        self.volume = VolumeConfig(settings.get('volume', {}))
        self.model = ModelConfig(settings.get('model', {}))
        self.network = NetworkConfig(settings.get('network', {}))
        self.optimizer = OptimizerConfig(settings.get('optimizer', {}))
        self.training = TrainingConfig(settings.get('training', {}))


CONFIG = Config.from_toml(os.path.join(os.path.dirname(__file__), 'conf', 'default.toml'))
