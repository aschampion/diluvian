# -*- coding: utf-8 -*-
"""Global configuration objects.

This module contains boilerplate configuration objects for storing and loading
configuration state.
"""


from __future__ import division

import os

import numpy as np
import pytoml as toml
import six


class BaseConfig(object):
    """Base class for configuration objects.

    String representation yields TOML that should parse back to a dictionary
    that will initialize the same configuration object.
    """
    def __str__(self):
        sanitized = {}
        for k, v in six.iteritems(self.__dict__):
            if isinstance(v, np.ndarray):
                sanitized[k] = v.tolist()
            else:
                sanitized[k] = v
        return toml.dumps(sanitized)

    __repr__ = __str__


class VolumeConfig(BaseConfig):
    """Configuration for the use of volumes.

    Attributes
    ----------
    resolution : sequence or ndarray of float
        Resolution to which volumes will be downsampled before processing.
    """
    def __init__(self, settings):
        self.resolution = np.array(settings.get('resolution', [1, 1, 1]))


class ModelConfig(BaseConfig):
    """Configuration for non-network aspects of the flood filling model.

    Attributes
    ----------
    input_fov_shape : sequence or ndarray of int
        Input field of view shape in voxels for each flood filling move.
    output_fov_shape : sequence or ndarray of int
        Output field of view shape in voxels for each flood filling move. Can
        not be larger than ``input_fov_shape``.
    output_fov_move_fraction : int
        Move size as a fraction of the output field of view shape.
    v_true, v_false : float
        Soft target values for in-object and out-of-object mask voxels,
        respectively.
    t_move : float
        Threshold mask probability in the move check plane to queue a move
        to that position.
    t_final : float, optional
        Threshold mask probability to produce the final segmentation. Defaults
        to ``t_move``.
    move_check_thickness : int
        Thickness of move check plane in voxels. Setting this greater than 1
        is useful to make moves more robust even if the move grid aligns with
        missing sections or image artifacts.
    training_subv_shape : sequence or ndarray of int, optional
        Shape of the subvolumes used during moving training.
    """
    def __init__(self, settings):
        self.input_fov_shape = np.array(settings.get('input_fov_shape', [17, 33, 33]))
        self.output_fov_shape = np.array(settings.get('output_fov_shape', [17, 33, 33]))
        self.output_fov_move_fraction = int(settings.get('output_fov_move_fraction', 4))
        self.v_true = float(settings.get('v_true', 0.95))
        self.v_false = float(settings.get('v_false', 0.05))
        self.t_move = float(settings.get('t_move', 0.9))
        self.t_final = float(settings.get('t_final', self.t_move))
        self.move_check_thickness = int(settings.get('move_check_thickness', 1))
        self.training_subv_shape = np.array(settings.get('training_subv_shape',
                                                         self.input_fov_shape + self.move_step * 2))

    @property
    def move_step(self):
        return (self.output_fov_shape - 1) // self.output_fov_move_fraction


class NetworkConfig(BaseConfig):
    """Configuration for the flood filling network architecture.

    Attributes
    ----------
    factory : str
        Module and function name for a factory method for creating the flood
        filling network. This allows a custom architecture to be provided
        without needing to modify diluvian.
    transpose : bool
        If true, any loaded networks will reverse the order of axes for both
        inputs and outputs. Data is assumed to be ZYX row-major, but old
        versions of diluvian used XYZ, so this is necessary to load old
        networks.
    num_modules : int
        Number of convolution modules to use, each module consisting of a skip
        link in parallel with two convolution layers.
    convolution_dim : sequence or ndarray of int
        Shape of the convolution for each layer.
    convolution_filters : int
        Number of convolution filters for each layer.
    initialization : str
        Name of the Keras initialization function to use for weight
        initialization of all layers.
    output_activation : str
        Name of the Keras activation function to use for the final network
        output.
    dropout_probability : float
        Probability for dropout layers. If zero, no dropout layers will be
        included.
    """
    def __init__(self, settings):
        self.factory = str(settings.get('factory'))
        self.transpose = bool(settings.get('transpose', False))
        self.num_modules = int(settings.get('num_modules', 8))
        self.convolution_dim = np.array(settings.get('convolution_dim', [3, 3, 3]))
        self.convolution_filters = int(settings.get('convolution_filters', 32))
        self.initialization = str(settings.get('initialization', 'glorot_uniform'))
        self.output_activation = str(settings.get('output_activation', 'sigmoid'))
        self.dropout_probability = float(settings.get('dropout_probability', 0.0))


class OptimizerConfig(BaseConfig):
    """Configuration for the network optimizer.

    Any settings dict entries passed to this initializer will be added as
    configuration attributes and passed to the optimizer initializer as keyword
    arguments.

    Attributes
    ----------
    klass : str
        Class name of the Keras optimizer to use.
    """
    def __init__(self, settings):
        for k, v in six.iteritems(settings):
            if k != 'klass':
                setattr(self, k, v)
        self.klass = str(settings.get('klass', 'SGD'))


class TrainingConfig(BaseConfig):
    """Configuration for model training.

    Attributes
    ----------
    num_gpus : int
        Number of GPUs to use for data-parallelism.
    num_workers : int
        Number of worker queues to use for generating training data.
    gpu_batch_size : int
        Per-GPU batch size. The effective batch size will be this times
        ``num_gpus``.
    training_size : int
        Number of samples to use for training **from each volume**.
    validation_size : int
        Number of samples to use for validation **from each volume**.
    static_train_epochs : int
        Number of epochs at the beginning of training where the model will not
        be allowed to move the FOV.
    total_epochs : int
        Maximum number of training epochs.
    reset_generators : bool
        Reset training generators after each epoch, so that the training
        examples at each epoch are identical.
    fill_factor_bins : sequence of float
        Bin boundaries for filling fractions. If provided, sample loss will be
        weighted to increase loss contribution from less-frequent bins.
        Otherwise all samples are weighted equally.
    partitions : dict
        Dictionary mapping volume name regexes to a sequence of int indicating
        number of volume partitions along each axis. Only one axis should be
        greater than 1. Each volume should match at most one regex.
    training_partition, validation_partition : dict
        Dictionaries mapping volume name regexes to a sequence of int indicating
        index of the partitions to use for training and validation,
        respectively. Each volume should match at most one regex.
    patience : int
        Number of epochs after the last minimal validation loss to terminate
        training.
    early_abort_epoch : int
        If provided, training will check at the end of this epoch
        whether validation loss is less than ``early_abort_loss``. If not,
        training will be aborted, and may be restarted with a new seed
        depending on CLI options. By default this is disabled.
    early_abort_loss : float
        See ``early_abort_epoch``.
    augment_mirrors : sequence of int
        Axes along which to mirror for data augmentation.
    augment_permute_axes : sequence of sequence of int
        Axis permutations to use for data augmentation.
    augment_missing_data : list of dict
        List of dictionaries with ``axis`` and ``prob`` keys, indicating
        an axis to perform data blanking along, and the probability to blank
        each plane in the axis, respectively.
    augment_noise : list of dict
        List of dictionaries with ``axis``, ``mul`` and `add`` keys, indicating
        an axis to perform independent Gaussian noise augmentation on, and the
        standard deviations of 1-mean multiplicative and 0-mean additive noise,
        respectively.
    augment_contrast : list of dict
        List of dictionaries with ``axis``, ``prob``, ``scaling_mean``,
        ``scaling_std``, ``center_mean`` and ``center_std`` keys. These
        specify the probability to alter the contrast of a section, the mean
        and standard deviation to draw from a normal distribution to scale
        contrast, and the mean and standard deviation to draw from a normal
        distribution to move the intensity center multiplicatively.
    """
    def __init__(self, settings):
        self.num_gpus = int(settings.get('num_gpus', 1))
        self.num_workers = int(settings.get('num_workers', 4))
        self.gpu_batch_size = int(settings.get('gpu_batch_size', 8))
        self.batch_size = self.num_gpus * self.gpu_batch_size
        self.training_size = int(settings.get('training_size', 256))
        self.validation_size = int(settings.get('validation_size', 256))
        self.static_train_epochs = int(settings.get('static_train_epochs', 10))
        self.total_epochs = int(settings.get('total_epochs', 100))
        self.reset_generators = bool(settings.get('reset_generators', False))
        self.fill_factor_bins = settings.get('fill_factor_bins', None)
        if self.fill_factor_bins is not None:
            self.fill_factor_bins = np.array(self.fill_factor_bins)
        self.partitions = settings.get('partitions', {'.*': [2, 1, 1]})
        self.training_partition = settings.get('training_partition', {'.*': [0, 0, 0]})
        self.validation_partition = settings.get('validation_partition', {'.*': [1, 0, 0]})
        self.patience = int(np.array(settings.get('patience', 10)))
        self.early_abort_epoch = settings.get('early_abort_epoch', None)
        self.early_abort_loss = settings.get('early_abort_loss', None)
        self.augment_mirrors = [int(x) for x in settings.get('augment_mirrors', [0, 1, 2])]
        self.augment_permute_axes = settings.get('augment_permute_axes', [[0, 2, 1]])
        self.augment_missing_data = settings.get('augment_missing_data', [{'axis': 0, 'prob': 0.01}])
        self.augment_noise = settings.get('augment_noise', [{'axis': 0, 'mul': 0.1, 'add': 0.1}])
        self.augment_contrast = settings.get(
                'augment_contrast',
                [{'axis': 0, 'prob': 0.05, 'scaling_mean': 0.5, 'scaling_std': 0.1,
                  'center_mean': 1.2, 'center_std': 0.2}])


class PostprocessingConfig(BaseConfig):
    """Configuration for segmentation processing after flood filling.

    Attributes
    ----------
    closing_shape : sequence or ndarray of int
        Shape of the structuring element for morphological closing, in voxels.
    """
    def __init__(self, settings):
        self.closing_shape = settings.get('closing_shape', None)


class Config(object):
    """A complete collection of configuration objects.

    Attributes
    ----------
    random_seed : int
        Seed for initializing the Python and NumPy random generators.
    """

    def __init__(self, settings_collection=None):
        if settings_collection is not None:
            settings = settings_collection[0].copy()
            for s in settings_collection:
                for c in s:
                    if c in settings and isinstance(settings[c], dict):
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
        self.postprocessing = PostprocessingConfig(settings.get('postprocessing', {}))

        self.random_seed = int(settings.get('random_seed', 0))

    def __str__(self):
        sanitized = {}
        for n, c in six.iteritems(self.__dict__):
            if not isinstance(c, BaseConfig):
                sanitized[n] = c
                continue
            sanitized[n] = {}
            for k, v in six.iteritems(c.__dict__):
                if isinstance(v, np.ndarray):
                    sanitized[n][k] = v.tolist()
                else:
                    sanitized[n][k] = v
        return toml.dumps(sanitized)

    def from_toml(self, *filenames):
        """Reinitializes this Config from a list of TOML configuration files.

        Existing settings are discarded. When multiple files are provided,
        configuration is overridden by later files in the list.

        Parameters
        ----------
        filenames : interable of str
            Filenames of TOML configuration files to load.
        """
        settings = []
        for filename in filenames:
            with open(filename, 'rb') as fin:
                settings.append(toml.load(fin))

        return self.__init__(settings)

    def to_toml(self, filename):
        with open(filename, 'w') as tomlfile:
            tomlfile.write(str(self))


CONFIG = Config()
CONFIG.from_toml(os.path.join(os.path.dirname(__file__), 'conf', 'default.toml'))
