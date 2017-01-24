import h5py
import itertools
import matplotlib.pyplot as plt
import numpy as np
import Queue

from keras.callbacks import Callback
from keras.layers import AveragePooling3D, Convolution3D, Input, merge
from keras.layers.core import Activation, Lambda, Merge
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras import backend as K

from scipy import stats

from progressbar import ProgressBar

import neuroglancer


DOWNSAMPLE = np.array((1, 1, 0))
# DOWNSAMPLE = np.array((0, 0, 0))
RESOLUTION = (8, 8, 40)
INPUT_SHAPE = np.array((65, 65, 13, 1))
NUM_MODULES = 8
CONV_X = CONV_Y = 5
CONV_Z = 3

T_MOVE = 0.75 # 0.9
MOVE_DELTA = (INPUT_SHAPE[0:3] - 1) / 4
TRAINING_FOV = INPUT_SHAPE[0:3] + (2 * MOVE_DELTA)

V_TRUE = 0.95
V_FALSE = 0.05

LEARNING_RATE = 0.01
BATCH_SIZE = 8
TRAINING_SIZE = 24
PRETRAIN_NUM_EPOCHS = 1
NUM_EPOCHS = 3


def add_convolution_module(model):
    model2 = Convolution3D(32, CONV_X, CONV_Y, CONV_Z,
                           activation='relu',
                           border_mode='same')(model)
    model2 = Convolution3D(32, CONV_X, CONV_Y, CONV_Z,
                           border_mode='same')(model2)
    model = merge([model, model2], mode='sum')
    # Note that the activation here differs from He et al 2016, as that
    # activation is not on the skip connection path.
    model = Activation('relu')(model)

    return model


class FloodFillRegion:
    ffrid = 0

    def __init__(self, image, target=None, seed_pos=None):
        self.ffrid = FloodFillRegion.ffrid
        FloodFillRegion.ffrid += 1
        self.queue = Queue.PriorityQueue()
        self.visited = set()
        self.image = image
        self.bounds = image.shape
        self.move_bounds = vox_to_pos(self.bounds) - 1
        self.mask = np.empty(self.bounds, dtype='float32')
        self.mask[:] = np.NAN
        self.target = target
        self.bias_against_merge = False
        if not seed_pos:
            seed_pos = np.floor_divide(self.move_bounds, 2) + 1
        self.queue.put((None, seed_pos))
        seed_vox = pos_to_vox(seed_pos)
        self.mask[tuple(seed_vox)] = V_TRUE
        # print 'FFR {0} with seed_pos: {1}'.format(self.ffrid, np.array_str(seed_pos))

    def add_mask(self, mask_block, mask_pos):
        new_moves = get_moves(mask_block)
        for move in new_moves:
            new_ctr = mask_pos + move['move']
            if np.any(np.greater_equal(new_ctr, self.move_bounds)) or np.any(new_ctr <= 1):
               continue
            if tuple(new_ctr) not in self.visited and move['v'] >= T_MOVE:
                self.visited.add(tuple(new_ctr))
                self.queue.put((-move['v'], tuple(new_ctr)))
                # print 'FFR {0} queuing {1} ({2})'.format(self.ffrid, np.array_str(new_ctr), move['v'])

        mask_origin = pos_to_vox(mask_pos) - (np.asarray(mask_block.shape) - 1) / 2

        if self.bias_against_merge:
            current_mask = self.mask[mask_origin[0]:mask_origin[0] + mask_block.shape[0],
                                     mask_origin[1]:mask_origin[1] + mask_block.shape[1],
                                     mask_origin[2]:mask_origin[2] + mask_block.shape[2]]
            current_mask[np.isnan(current_mask) | current_mask > 0.5 | mask_block < current_mask] = mask_block
            self.mask[mask_origin[0]:mask_origin[0] + mask_block.shape[0],
                      mask_origin[1]:mask_origin[1] + mask_block.shape[1],
                      mask_origin[2]:mask_origin[2] + mask_block.shape[2]] = current_mask
        else:
            self.mask[mask_origin[0]:mask_origin[0] + mask_block.shape[0],
                      mask_origin[1]:mask_origin[1] + mask_block.shape[1],
                      mask_origin[2]:mask_origin[2] + mask_block.shape[2]] = mask_block

    def get_next_block(self):
        next_pos = np.asarray(self.queue.get()[1])
        # print 'FFR {0} dequeuing {1}'.format(self.ffrid, np.array_str(next_pos))
        next_vox = pos_to_vox(next_pos)
        margin = (INPUT_SHAPE[0:3] - 1) / 2
        block_min = next_vox - margin
        block_max = next_vox + margin + 1
        image_block = self.image[block_min[0]:block_max[0],
                                 block_min[1]:block_max[1],
                                 block_min[2]:block_max[2]]
        mask_block = self.mask[block_min[0]:block_max[0],
                               block_min[1]:block_max[1],
                               block_min[2]:block_max[2]]

        mask_block[np.isnan(mask_block)] = V_FALSE

        if self.target is not None:
            target_block = self.target[block_min[0]:block_max[0],
                                       block_min[1]:block_max[1],
                                       block_min[2]:block_max[2]]
        else:
            target_block = None

        assert image_block.shape == tuple(INPUT_SHAPE[0:3]), 'Image wrong size: {}'.format(image_block.shape)
        assert mask_block.shape == tuple(INPUT_SHAPE[0:3]), 'Mask wrong size: {}'.format(mask_block.shape)
        return {'image': image_block,
                'mask': mask_block,
                'target': target_block,
                'position': next_pos}

    def get_viewer(self):
        viewer = neuroglancer.Viewer(voxel_size=list(RESOLUTION))
        viewer.add(np.transpose(self.image),
                   name='Image')
        viewer.add(np.transpose(self.target),
                   name='Mask Target',
                   shader="""
void main() {
  emitRGBA(vec4(toNormalized(getDataValue(0)), 0, 0, toNormalized(getDataValue(0))));
}
""")
        viewer.add(np.transpose(self.mask),
                   name='Mask Output',
                   shader="""
void main() {
  emitRGBA(vec4(0, toNormalized(getDataValue(0)), 0, toNormalized(getDataValue(0))));
}
""")
        return viewer


def vox_to_pos(vox):
    return np.divide(vox, MOVE_DELTA)


def pos_to_vox(pos):
    return pos * MOVE_DELTA


def get_moves(mask):
    moves = []
    ctr = (np.asarray(mask.shape) - 1) / 2 + 1
    for move in map(np.array, [(1, 0, 0), (-1, 0, 0),
                 (0, 1, 0), (0, -1, 0),
                 (0, 0, 1), (0, 0, -1)]):
        moves.append({'move': move,
                      'v': mask[ctr[0] - (-2 * max(move[0], 0) + 1) * MOVE_DELTA[0]:
                                ctr[0] + (+2 * min(move[0], 0) + 1) * MOVE_DELTA[0] + 1,
                                ctr[1] - (-2 * max(move[1], 0) + 1) * MOVE_DELTA[1]:
                                ctr[1] + (+2 * min(move[1], 0) + 1) * MOVE_DELTA[1] + 1,
                                ctr[2] - (-2 * max(move[2], 0) + 1) * MOVE_DELTA[2]:
                                ctr[2] + (+2 * min(move[2], 0) + 1) * MOVE_DELTA[2] + 1].max()})
    return moves


def simple_training_generator(orig_file, image_group, image_dataset, label_group, label_dataset, batch_size, training_size):
    f = h5py.File(orig_file, 'r')
    zoom = np.exp2(DOWNSAMPLE).astype('int64')
    region_size_zoom = INPUT_SHAPE[0:3]
    region_size_orig = np.multiply(region_size_zoom, zoom)
    image_data = f[image_group][image_dataset]
    label_data = f[label_group][label_dataset]
    ctr_min = np.floor_divide(region_size_orig, 2)
    # HDF5 coordinates are z, y, x
    ctr_max = (np.flipud(np.array(image_data.shape)) - (np.floor_divide(region_size_orig, 2) + 1))

    mask_input = np.full(INPUT_SHAPE, V_FALSE, dtype='float32')
    mask_input[tuple(np.array(mask_input.shape) / 2)] = V_TRUE
    mask_input = np.tile(mask_input, (batch_size, 1, 1, 1, 1))

    def pad_dims(x):
        return np.expand_dims(np.expand_dims(x, x.ndim), 0)

    np.random.seed(0)
    sample_num = 0
    while 1:
        if sample_num >= training_size:
            np.random.seed(0)
            sample_num = 0

        batch_image_input = None
        batch_mask_target = None

        for _ in range(0, batch_size):
            ctr = tuple(np.random.randint(ctr_min[n], ctr_max[n]) for n in range(0, 3))
            subvol = ((ctr[2] - ctr_min[2], ctr[2] + ctr_min[2] + (region_size_orig[2] % 2)),
                      (ctr[1] - ctr_min[1], ctr[1] + ctr_min[1] + (region_size_orig[1] % 2)),
                      (ctr[0] - ctr_min[0], ctr[0] + ctr_min[0] + (region_size_orig[0] % 2)))
            image_subvol = image_data[subvol[0][0]:subvol[0][1],
                                      subvol[1][0]:subvol[1][1],
                                      subvol[2][0]:subvol[2][1]]
            label_subvol = label_data[subvol[0][0]:subvol[0][1],
                                      subvol[1][0]:subvol[1][1],
                                      subvol[2][0]:subvol[2][1]]

            image_subvol = np.transpose(image_subvol.astype('float32')) / 256.0
            label_subvol = np.transpose(label_subvol)

            if np.count_nonzero(DOWNSAMPLE):
                image_subvol = image_subvol.reshape([region_size_zoom[0], zoom[0],
                                                     region_size_zoom[1], zoom[1],
                                                     region_size_zoom[2], zoom[2]]).mean(5).mean(3).mean(1)
                label_subvol = label_subvol.reshape([region_size_zoom[0], zoom[0],
                                                     region_size_zoom[1], zoom[1],
                                                     region_size_zoom[2], zoom[2]])
                label_subvol = stats.mode(label_subvol, 5)[0]
                label_subvol = stats.mode(label_subvol, 3)[0]
                label_subvol = np.squeeze(stats.mode(label_subvol, 1)[0])

            assert image_subvol.shape == tuple(region_size_zoom), 'Image wrong size: {}'.format(image_subvol.shape)
            assert label_subvol.shape == tuple(region_size_zoom), 'Labels wrong size: {}'.format(label_subvol.shape)

            label_id = label_subvol[tuple(np.array(label_subvol.shape) / 2)]
            label_mask = label_subvol == label_id
            f_a = np.count_nonzero(label_mask) / float(label_mask.size)
            mask_target = np.full_like(label_subvol, V_FALSE, dtype='float32')
            mask_target[label_mask] = V_TRUE
            # print 'Yielding (' + ','.join(map(str, ctr)) + ') Label ID: ' + str(label_id) + ' f_a: {:.1%}'.format(f_a)

            image_input = pad_dims(image_subvol)
            mask_target = pad_dims(mask_target)

            batch_image_input = np.concatenate((batch_image_input, image_input)) if batch_image_input is not None else image_input
            batch_mask_target = np.concatenate((batch_mask_target, mask_target)) if batch_mask_target is not None else mask_target

        sample_num += batch_size
        yield ({'image_input': batch_image_input,
                'mask_input': mask_input},
               {'mask_output': batch_mask_target})


def moving_training_generator(orig_file, image_group, image_dataset, label_group, label_dataset, batch_size, training_size, callback_kludge):
    f = h5py.File(orig_file, 'r')
    zoom = np.exp2(DOWNSAMPLE).astype('int64')
    region_size_zoom = TRAINING_FOV[0:3]
    region_size_orig = np.multiply(region_size_zoom, zoom)
    image_data = f[image_group][image_dataset]
    label_data = f[label_group][label_dataset]
    ctr_min = np.floor_divide(region_size_orig, 2)
    # HDF5 coordinates are z, y, x
    ctr_max = (np.flipud(np.array(image_data.shape)) - (np.floor_divide(region_size_orig, 2) + 1))

    def pad_dims(x):
        return np.expand_dims(np.expand_dims(x, x.ndim), 0)

    regions = [None] * batch_size
    region_pos = [None] * batch_size

    np.random.seed(0)
    sample_num = 0
    while 1:
        if sample_num >= training_size:
            np.random.seed(0)
            sample_num = 0

        # Before clearing last batches, reuse them to predict mask outputs
        # for move training. Add mask outputs to regions.
        active_regions = [n for n, region in enumerate(regions) if region is not None]
        if active_regions and callback_kludge['outputs'] is not None:
            # mask_updates = model.predict({'image_input': batch_image_input,
            #                               'mask_input': batch_mask_input})
            # for n in active_regions:
            #     regions[n].add_mask(mask_updates[n, :, :, :, 0], region_pos[n])
            for n in active_regions:
                assert np.array_equal(callback_kludge['inputs']['image_input'][n, 0, 0, :, 0], batch_image_input[n, 0, 0, :, 0])
                regions[n].add_mask(callback_kludge['outputs'][n, :, :, :, 0], region_pos[n])

        batch_image_input = None
        batch_mask_input = None
        batch_mask_target = None

        for r, region in enumerate(regions):
            if region is None or region.queue.empty():
                ctr = tuple(np.random.randint(ctr_min[n], ctr_max[n]) for n in range(0, 3))
                subvol = ((ctr[2] - ctr_min[2], ctr[2] + ctr_min[2] + (region_size_orig[2] % 2)),
                          (ctr[1] - ctr_min[1], ctr[1] + ctr_min[1] + (region_size_orig[1] % 2)),
                          (ctr[0] - ctr_min[0], ctr[0] + ctr_min[0] + (region_size_orig[0] % 2)))
                image_subvol = image_data[subvol[0][0]:subvol[0][1],
                                          subvol[1][0]:subvol[1][1],
                                          subvol[2][0]:subvol[2][1]]
                label_subvol = label_data[subvol[0][0]:subvol[0][1],
                                          subvol[1][0]:subvol[1][1],
                                          subvol[2][0]:subvol[2][1]]

                image_subvol = np.transpose(image_subvol.astype('float32')) / 256.0
                label_subvol = np.transpose(label_subvol)

                if np.count_nonzero(DOWNSAMPLE):
                    image_subvol = image_subvol.reshape([region_size_zoom[0], zoom[0],
                                                         region_size_zoom[1], zoom[1],
                                                         region_size_zoom[2], zoom[2]]).mean(5).mean(3).mean(1)
                    label_subvol = label_subvol.reshape([region_size_zoom[0], zoom[0],
                                                         region_size_zoom[1], zoom[1],
                                                         region_size_zoom[2], zoom[2]])
                    label_subvol = stats.mode(label_subvol, 5)[0]
                    label_subvol = stats.mode(label_subvol, 3)[0]
                    label_subvol = np.squeeze(stats.mode(label_subvol, 1)[0])

                assert image_subvol.shape == tuple(region_size_zoom), 'Image wrong size: {}'.format(image_subvol.shape)
                assert label_subvol.shape == tuple(region_size_zoom), 'Labels wrong size: {}'.format(label_subvol.shape)

                label_id = label_subvol[tuple(np.array(label_subvol.shape) / 2)]
                label_mask = label_subvol == label_id
                f_a = np.count_nonzero(label_mask) / float(label_mask.size)
                mask_target = np.full_like(label_subvol, V_FALSE, dtype='float32')
                mask_target[label_mask] = V_TRUE

                regions[r] = FloodFillRegion(image_subvol, mask_target)
                region = regions[r]

            block_data = region.get_next_block()

            image_input = pad_dims(block_data['image'])
            mask_input = pad_dims(block_data['mask'])
            mask_target = pad_dims(block_data['target'])
            region_pos[r] = block_data['position']

            batch_image_input = np.concatenate((batch_image_input, image_input)) if batch_image_input is not None else image_input
            batch_mask_input = np.concatenate((batch_mask_input, mask_input)) if batch_mask_input is not None else mask_input
            batch_mask_target = np.concatenate((batch_mask_target, mask_target)) if batch_mask_target is not None else mask_target

        sample_num += batch_size
        inputs = {'image_input': batch_image_input,
                'mask_input': batch_mask_input}
        callback_kludge['inputs'] = inputs
        yield (inputs,
               {'mask_output': batch_mask_target})


def make_network():
    image_input = Input(shape=tuple(INPUT_SHAPE), dtype='float32', name='image_input')
    mask_input = Input(shape=tuple(INPUT_SHAPE), dtype='float32', name='mask_input')
    ffn = merge([image_input, mask_input], mode='concat')

    # Convolve and activate before beginning the skip connection modules,
    # as discussed in the Appendix of He et al 2016.
    ffn = Convolution3D(32, CONV_X, CONV_Y, CONV_Z,
                        activation='relu',
                        border_mode='same')(ffn)

    for _ in range(0, NUM_MODULES):
        ffn = add_convolution_module(ffn)

    mask_output = Convolution3D(1, 1, 1, 1, name='mask_output', activation='hard_sigmoid')(ffn)
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


class PredictionCopy(Callback):
    def __init__(self, kludge):
        self.kludge = kludge

    def on_batch_end(self, batch, logs={}):
        if self.kludge['inputs']:
            self.kludge['outputs'] = self.model.predict(self.kludge['inputs'])


def extend_keras_history(a, b):
    a.epoch.extend(b.epoch)
    for k, v in b.history.items():
        a.history.setdefault(k, []).extend(v)


def main():
    ffn = make_network()

    # Pre-train
    training_data = simple_training_generator('/home/championa/code/catsop/cremi-export/orig/sample_A_20160501.hdf',
                                       '/volumes',
                                       'raw',
                                       'volumes/labels',
                                       'neuron_ids',
                                       BATCH_SIZE,
                                       TRAINING_SIZE)
    history = ffn.fit_generator(training_data,
                                samples_per_epoch=TRAINING_SIZE,
                                nb_epoch=PRETRAIN_NUM_EPOCHS)

    # Moving training
    kludge = {'inputs': None, 'outputs': None}
    cb = PredictionCopy(kludge)
    training_data = moving_training_generator('/home/championa/code/catsop/cremi-export/orig/sample_A_20160501.hdf',
                                       '/volumes',
                                       'raw',
                                       'volumes/labels',
                                       'neuron_ids',
                                       BATCH_SIZE,
                                       TRAINING_SIZE,
                                       kludge)
    moving_history = ffn.fit_generator(training_data,
                                samples_per_epoch=TRAINING_SIZE,
                                nb_epoch=NUM_EPOCHS,
                                initial_epoch=PRETRAIN_NUM_EPOCHS,
                                max_q_size=1,
                                nb_worker=1,
                                callbacks=[cb])
    extend_keras_history(history, moving_history)

    # for _ in itertools.islice(training_data, 12):
    #     continue
    dupe_data = simple_training_generator('/home/championa/code/catsop/cremi-export/orig/sample_A_20160501.hdf',
                                       '/volumes',
                                       'raw',
                                       'volumes/labels',
                                       'neuron_ids',
                                       BATCH_SIZE,
                                       TRAINING_SIZE)
    viz_ex = itertools.islice(dupe_data, 1)

    for inputs, targets in viz_ex:
        viewer = neuroglancer.Viewer(voxel_size=list(RESOLUTION))
        viewer.add(np.transpose(inputs['image_input'][0, :, :, :, 0]),
                   name='Image')
        viewer.add(np.transpose(inputs['mask_input'][0, :, :, :, 0]),
                   name='Mask Input',
                   shader="""
void main() {
  emitRGBA(vec4(0, 0, toNormalized(getDataValue(0)), toNormalized(getDataValue(0))));
}
""")
        viewer.add(np.transpose(targets['mask_output'][0, :, :, :, 0]),
                   name='Mask Target',
                   shader="""
void main() {
  emitRGBA(vec4(toNormalized(getDataValue(0)), 0, 0, toNormalized(getDataValue(0))));
}
""")
        output = ffn.predict(inputs)
        viewer.add(np.transpose(output[0, :, :, :, 0]),
                   name='Mask Output',
                   shader="""
void main() {
  emitRGBA(vec4(0, toNormalized(getDataValue(0)), 0, toNormalized(getDataValue(0))));
}
""")
        print(viewer)
    plot_history(history)
    return history

if __name__ == "__main__":
    main()
