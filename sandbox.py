import h5py
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import AveragePooling3D, Convolution3D, Input, merge
from keras.layers.core import Activation, Lambda, Merge
from keras.models import Model, Sequential
from keras.optimizers import SGD

from progressbar import ProgressBar


DOWNSAMPLE = (2, 2, 0)
RESOLUTION = (16, 16, 40)
INPUT_SHAPE = (33, 33, 13, 1)
NUM_MODULES = 8

T_MOVE = 0.9
STEP_DELTA = (8, 8, 3)
TRAINING_FOV = tuple(map(sum, zip(INPUT_SHAPE, map((2).__mul__, STEP_DELTA))))

V_TRUE = 0.95
V_FALSE = 0.05

LEARNING_RATE = 0.001
BATCH_SIZE = 4


def add_convolution_module(model):
    model2 = Convolution3D(32, 3, 3, 3,
                           activation='relu',
                           border_mode='same')(model)
    model2 = Convolution3D(32, 3, 3, 3,
                           border_mode='same')(model2)
    model = merge([model, model2], mode='sum')
    # Note that the activation here differs from He et al 2016, as that
    # activation is not on the skip connection path.
    model = Activation('relu')(model)

    return model


def training_generator(orig_file, image_group, image_dataset, label_group, label_dataset):
    f = h5py.File(orig_file, 'r')
    image_data = f[image_group][image_dataset]
    label_data = f[label_group][label_dataset]
    ctr_min = tuple(i // 2 for i in INPUT_SHAPE)
    # HDF5 coordinates are z, y, x
    ctr_max = tuple(image_data.shape[2 - idx] - (i // 2 + 1) for idx, i, in enumerate(INPUT_SHAPE))

    mask_input = np.full(INPUT_SHAPE, V_FALSE, dtype='float32')
    mask_input[ctr_min[0], ctr_min[1], ctr_min[2]] = V_TRUE
    mask_input = np.expand_dims(mask_input, 0)

    def pad_dims(x):
        # return np.expand_dims(x, x.ndim)
        return np.expand_dims(np.expand_dims(x, x.ndim), 0)

    np.random.seed(0)
    while 1:
        ctr = tuple(np.random.randint(ctr_min[i], ctr_max[i]) for i in range(0, 3))
        subvol = ((ctr[2] - ctr_min[2], ctr[2] + ctr_min[2] + 1),
                  (ctr[1] - ctr_min[1], ctr[1] + ctr_min[1] + 1),
                  (ctr[0] - ctr_min[0], ctr[0] + ctr_min[0] + 1))
        image_subvol = image_data[subvol[0][0]:subvol[0][1],
                                  subvol[1][0]:subvol[1][1],
                                  subvol[2][0]:subvol[2][1]]
        label_subvol = label_data[subvol[0][0]:subvol[0][1],
                                  subvol[1][0]:subvol[1][1],
                                  subvol[2][0]:subvol[2][1]]
        # label_id = label_data[ctr[2], ctr[1], ctr[0]]
        label_id = label_subvol[ctr_min[2], ctr_min[1], ctr_min[0]]
        label_mask = label_subvol == label_id
        f_a = np.count_nonzero(label_mask) / float(label_mask.size)
        mask_target = np.full_like(label_subvol, V_FALSE, dtype='float32')
        mask_target[label_mask] = V_TRUE
        # print 'Yielding (' + ','.join(map(str, ctr)) + ') Label ID: ' + str(label_id) + ' f_a: {:.1%}'.format(f_a)

        yield ({'image_input': pad_dims(np.transpose(image_subvol)),
                'mask_input': mask_input},
               {'mask_output': pad_dims(np.transpose(mask_target))})



def make_network():
    image_input = Input(shape=INPUT_SHAPE, dtype='float32', name='image_input')
    mask_input = Input(shape=INPUT_SHAPE, dtype='float32', name='mask_input')
    ffn = merge([image_input, mask_input], mode='concat')

    # Convolve and activate before beginning the skip connection modules,
    # as discussed in the Appendix of He et al 2016.
    ffn = Convolution3D(32, 3, 3, 3,
                        activation='relu',
                        border_mode='same')(ffn)

    for _ in range(0, NUM_MODULES):
        ffn = add_convolution_module(ffn)

    mask_output = Convolution3D(1, 1, 1, 1, name='mask_output')(ffn)
    ffn = Model(input=[image_input, mask_input], output=[mask_output])
    ffn.compile(loss='binary_crossentropy',
                optimizer=SGD(lr=LEARNING_RATE))

    return ffn


def plot_history(history):
    # plt.plot(history.history['acc'])
    # # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def main():
    ffn = make_network()
    history = ffn.fit_generator(training_generator('/home/championa/code/catsop/cremi-export/orig/sample_A_20160501.hdf',
                                                   '/volumes',
                                                   'raw',
                                                   'volumes/labels',
                                                   'neuron_ids'),
                                samples_per_epoch=10,
                                nb_epoch=5)
    plot_history(history)
    return history

if __name__ == "__main__":
    main()
