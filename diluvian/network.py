# -*- coding: utf-8 -*-
"""Flood-fill network creation and compilation using Keras."""


import inspect

from keras.layers import (
        Convolution3D,
        Input,
        merge,
        )
from keras.layers.core import Activation
from keras.models import Model
import keras.optimizers


def make_flood_fill_network(fov_shape, network_config):
    image_input = Input(shape=tuple(fov_shape) + (1,), dtype='float32', name='image_input')
    mask_input = Input(shape=tuple(fov_shape) + (1,), dtype='float32', name='mask_input')
    ffn = merge([image_input, mask_input], mode='concat')

    # Convolve and activate before beginning the skip connection modules,
    # as discussed in the Appendix of He et al 2016.
    ffn = Convolution3D(network_config.convolution_filters,
                        network_config.convolution_dim[0],
                        network_config.convolution_dim[1],
                        network_config.convolution_dim[2],
                        init=network_config.initialization,
                        activation='relu',
                        border_mode='same')(ffn)

    for _ in range(0, network_config.num_modules):
        ffn = add_convolution_module(ffn, network_config)

    mask_output = Convolution3D(1,
                                network_config.convolution_dim[0],
                                network_config.convolution_dim[1],
                                network_config.convolution_dim[2],
                                init=network_config.initialization,
                                border_mode='same',
                                name='mask_output',
                                activation=network_config.output_activation)(ffn)
    ffn = Model(input=[image_input, mask_input], output=[mask_output])

    return ffn


def add_convolution_module(model, network_config):
    model2 = Convolution3D(network_config.convolution_filters,
                           network_config.convolution_dim[0],
                           network_config.convolution_dim[1],
                           network_config.convolution_dim[2],
                           init=network_config.initialization,
                           activation='relu',
                           border_mode='same')(model)
    model2 = Convolution3D(network_config.convolution_filters,
                           network_config.convolution_dim[0],
                           network_config.convolution_dim[1],
                           network_config.convolution_dim[2],
                           init=network_config.initialization,
                           border_mode='same')(model2)
    model = merge([model, model2], mode='sum')
    # Note that the activation here differs from He et al 2016, as that
    # activation is not on the skip connection path. However, this is not
    # likely to be important, see:
    # http://torch.ch/blog/2016/02/04/resnets.html
    # https://github.com/gcr/torch-residual-networks
    model = Activation('relu')(model)

    return model


def compile_network(model, optimizer_config):
    optimizer_klass = getattr(keras.optimizers, optimizer_config.klass)
    optimizer_kwargs = inspect.getargspec(optimizer_klass.__init__)[0]
    optimizer_kwargs = {k: v for k, v in optimizer_config.__dict__.iteritems() if k in optimizer_kwargs}
    optimizer = optimizer_klass(**optimizer_kwargs)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer)
