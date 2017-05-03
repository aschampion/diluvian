"""Parallelize training a Keras model across multiple GPUs.

Originally by Alex Kouzemtchenko, taken from:
https://github.com/kuza55/keras-extras/utils/multi_gpu.py

License: Apache
"""
from keras import backend as K
from keras.layers.core import Lambda
from keras.layers.merge import concatenate
from keras.models import Model

import tensorflow as tf

def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        # Adapted from:
        # https://github.com/fchollet/keras/issues/2436#issuecomment-291874528
        sh = K.shape(data)
        L = sh[0] / parts
        if idx == parts - 1:
            return data[idx*L:]
        return data[idx*L:(idx+1)*L]

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(concatenate(outputs, axis=0))

        # From https://github.com/kuza55/keras-extras/issues/3#issuecomment-264408864
        new_model = Model(inputs=model.inputs, outputs=merged)
        func_type = type(model.save)

        # monkeypatch the save to save just the underlying model
        def new_save(_, *args, **kwargs):
            model.save(*args, **kwargs)
        new_model.save = func_type(new_save, new_model)

        return new_model

