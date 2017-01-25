import h5py
import itertools
import matplotlib.pyplot as plt
import numpy as np
import Queue

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import AveragePooling3D, Convolution3D, Input, merge
from keras.layers.core import Activation, Lambda, Merge
from keras.models import load_model, Model, Sequential
from keras.optimizers import SGD
from keras import backend as K

from scipy import stats

from progressbar import ProgressBar

import neuroglancer


# DOWNSAMPLE = np.array((1, 1, 0))
DOWNSAMPLE = np.array((0, 0, 0))
# RESOLUTION = (8, 8, 40)
RESOLUTION = (4, 4, 40)
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
VALIDATION_SIZE = 16
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

    def fill(self, model):
        while not self.queue.empty():
            block_data = self.get_next_block()

            image_input = pad_dims(block_data['image'])
            mask_input = pad_dims(block_data['mask'])

            output = model.predict({'image_input': image_input,
                                    'mask_input': mask_input})
            self.add_mask(output[0, :, :, :, 0], block_data['position'])

    def get_viewer(self):
        viewer = neuroglancer.Viewer(voxel_size=list(RESOLUTION))
        viewer.add(np.transpose(self.image),
                   name='Image')
        viewer.add(np.transpose(self.target),
                   name='Mask Target',
                   shader=get_color_shader(0))
        viewer.add(np.transpose(self.mask),
                   name='Mask Output',
                   shader=get_color_shader(1))
        return viewer


def vox_to_pos(vox):
    return np.floor_divide(vox, MOVE_DELTA)


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


def pad_dims(x):
    """Add single-dimensions to the beginning and end of an array."""
    return np.expand_dims(np.expand_dims(x, x.ndim), 0)


class HDF5Volume:
    def __init__(self, orig_file, image_dataset, label_dataset):
        self.file = h5py.File(orig_file, 'r')
        self.image_data = self.file[image_dataset]
        self.label_data = self.file[label_dataset]

    def simple_training_generator(self, subvolume_size, batch_size, training_size, partition=None):
        subvolumes = self.SubvolumeGenerator(self, subvolume_size, DOWNSAMPLE, partition)

        mask_input = np.full(np.append(subvolume_size, (1,)), V_FALSE, dtype='float32')
        mask_input[tuple(np.array(mask_input.shape) / 2)] = V_TRUE
        mask_input = np.tile(mask_input, (batch_size, 1, 1, 1, 1))

        sample_num = 0
        while 1:
            if sample_num >= training_size:
                subvolumes.reset()
                sample_num = 0

            batch_image_input = None
            batch_mask_target = None

            for _ in range(0, batch_size):
                subvolume = subvolumes.next()

                image_input = pad_dims(subvolume['image'])
                mask_target = pad_dims(subvolume['mask_target'])

                batch_image_input = np.concatenate((batch_image_input, image_input)) if batch_image_input is not None else image_input
                batch_mask_target = np.concatenate((batch_mask_target, mask_target)) if batch_mask_target is not None else mask_target

            sample_num += batch_size
            yield ({'image_input': batch_image_input,
                    'mask_input': mask_input},
                   {'mask_output': batch_mask_target})

    def moving_training_generator(self, subvolume_size, batch_size, training_size, callback_kludge, partition=None):
        subvolumes = self.SubvolumeGenerator(self, subvolume_size, DOWNSAMPLE, partition)

        regions = [None] * batch_size
        region_pos = [None] * batch_size

        sample_num = 0
        while 1:
            if sample_num >= training_size:
                subvolumes.reset()
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
                    subvolume = subvolumes.next()

                    regions[r] = FloodFillRegion(subvolume['image'], subvolume['mask_target'])
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

    def region_generator(self, subvolume_size, partition=None):
        subvolumes = self.SubvolumeGenerator(self, subvolume_size, DOWNSAMPLE, partition)

        while 1:
            subvolume = subvolumes.next()
            region = FloodFillRegion(subvolume['image'], subvolume['mask_target'])
            yield region


    class SubvolumeGenerator:
        def __init__(self, volume, size_zoom, downsample, partition=None):
            if partition is None:
                partition = (np.array((1, 1, 1)), np.array((0, 0, 0)))
            self.volume = volume
            self.partition = partition
            self.zoom = np.exp2(downsample).astype('int64')
            self.size_zoom = size_zoom
            self.size_orig = np.multiply(self.size_zoom, self.zoom)
            self.margin = np.floor_divide(self.size_orig, 2)
            # HDF5 coordinates are z, y, x
            self.partition_size = np.floor_divide(np.flipud(np.array(self.volume.image_data.shape)), self.partition[0])
            self.ctr_min = np.multiply(self.partition_size, self.partition[1]) + self.margin
            self.ctr_max = np.multiply(self.partition_size, self.partition[1] + 1) - self.margin - 1
            self.random = np.random.RandomState(0)

        def __iter__(self):
            return self

        def reset(self):
            self.random.seed(0)

        def next(self):
            ctr = tuple(self.random.randint(self.ctr_min[n], self.ctr_max[n]) for n in range(0, 3))
            subvol = ((ctr[2] - self.margin[2], ctr[2] + self.margin[2] + (self.size_orig[2] % 2)),
                      (ctr[1] - self.margin[1], ctr[1] + self.margin[1] + (self.size_orig[1] % 2)),
                      (ctr[0] - self.margin[0], ctr[0] + self.margin[0] + (self.size_orig[0] % 2)))
            image_subvol = self.volume.image_data[subvol[0][0]:subvol[0][1],
                                      subvol[1][0]:subvol[1][1],
                                      subvol[2][0]:subvol[2][1]]
            label_subvol = self.volume.label_data[subvol[0][0]:subvol[0][1],
                                      subvol[1][0]:subvol[1][1],
                                      subvol[2][0]:subvol[2][1]]

            image_subvol = np.transpose(image_subvol.astype('float32')) / 256.0
            label_subvol = np.transpose(label_subvol)
            label_id = label_subvol[tuple(np.array(label_subvol.shape) / 2)]
            label_mask = label_subvol == label_id

            if np.any(self.zoom > 1):
                image_subvol = image_subvol.reshape([self.size_zoom[0], self.zoom[0],
                                                     self.size_zoom[1], self.zoom[1],
                                                     self.size_zoom[2], self.zoom[2]]).mean(5).mean(3).mean(1)
                label_mask = label_mask.reshape([self.size_zoom[0], self.zoom[0],
                                                 self.size_zoom[1], self.zoom[1],
                                                 self.size_zoom[2], self.zoom[2]]).all(5).all(3).all(1)
                # A higher fidelity alternative would be to use the mode label
                # for each downsample block. However, this is prohibitively
                # slow using the scipy code preserved below as an example:
                # label_subvol = label_subvol.reshape([region_size_zoom[0], zoom[0],
                #                                      region_size_zoom[1], zoom[1],
                #                                      region_size_zoom[2], zoom[2]])
                # label_subvol = stats.mode(label_subvol, 5)[0]
                # label_subvol = stats.mode(label_subvol, 3)[0]
                # label_subvol = np.squeeze(stats.mode(label_subvol, 1)[0])

            assert image_subvol.shape == tuple(self.size_zoom), 'Image wrong size: {}'.format(image_subvol.shape)
            assert label_mask.shape == tuple(self.size_zoom), 'Labels wrong size: {}'.format(label_mask.shape)

            f_a = np.count_nonzero(label_mask) / float(label_mask.size)
            mask_target = np.full_like(label_mask, V_FALSE, dtype='float32')
            mask_target[label_mask] = V_TRUE

            return {'image': image_subvol, 'mask_target': mask_target}


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
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
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


def get_color_shader(channel):
    value_str = 'toNormalized(getDataValue(0))'
    channels = ['0', '0', '0', value_str]
    channels[channel] = value_str
    shader = """
void main() {{
  emitRGBA(vec4({}));
}}
""".format(', '.join(channels))
    return shader


def fill_region_from_model(model_file):
    hdf5_file = '/home/championa/code/catsop/cremi-export/orig/sample_A_20160501.hdf'
    image_dataset = 'volumes/raw'
    label_dataset = 'volumes/labels/neuron_ids'

    volume = HDF5Volume(hdf5_file, image_dataset, label_dataset)

    regions = volume.region_generator(TRAINING_FOV * 4)

    model = load_model(model_file)

    for region in regions:
        region.fill(model)
        viewer = region.get_viewer()
        print viewer
        s = raw_input("Press Enter to continue, q to quit...")
        if s == 'q':
            break


def main():
    ffn = make_network()

    hdf5_file = '/home/championa/code/catsop/cremi-export/orig/sample_A_20160501.hdf'
    image_dataset = 'volumes/raw'
    label_dataset = 'volumes/labels/neuron_ids'

    partitions = np.array((1, 1, 2))

    volume = HDF5Volume(hdf5_file, image_dataset, label_dataset)
    validation_data = volume.simple_training_generator(INPUT_SHAPE[0:3],
                                       BATCH_SIZE,
                                       VALIDATION_SIZE,
                                       (partitions, np.array((0, 0, 1))))

    # Pre-train
    training_data = volume.simple_training_generator(INPUT_SHAPE[0:3],
                                                     BATCH_SIZE,
                                                     TRAINING_SIZE,
                                                     (partitions, np.array((0, 0, 0))))
    history = ffn.fit_generator(training_data,
                                samples_per_epoch=TRAINING_SIZE,
                                nb_epoch=PRETRAIN_NUM_EPOCHS,
                                validation_data=validation_data,
                                nb_val_samples=VALIDATION_SIZE)

    # Moving training
    kludge = {'inputs': None, 'outputs': None}
    cb = PredictionCopy(kludge)
    checkpoint = ModelCheckpoint('weights.hdf5', save_best_only=True)
    early_stop = EarlyStopping(patience=50)
    training_data = volume.moving_training_generator(TRAINING_FOV,
                                       BATCH_SIZE,
                                       TRAINING_SIZE,
                                       kludge,
                                       (partitions, np.array((0, 0, 0))))
    moving_history = ffn.fit_generator(training_data,
                                samples_per_epoch=TRAINING_SIZE,
                                nb_epoch=NUM_EPOCHS,
                                initial_epoch=PRETRAIN_NUM_EPOCHS,
                                max_q_size=1,
                                nb_worker=1,
                                callbacks=[cb, checkpoint, early_stop],
                                validation_data=validation_data,
                                nb_val_samples=VALIDATION_SIZE)
    extend_keras_history(history, moving_history)

    # for _ in itertools.islice(training_data, 12):
    #     continue
    dupe_data = volume.simple_training_generator(INPUT_SHAPE[0:3],
                                       BATCH_SIZE,
                                       TRAINING_SIZE)
    viz_ex = itertools.islice(dupe_data, 1)

    for inputs, targets in viz_ex:
        viewer = neuroglancer.Viewer(voxel_size=list(RESOLUTION))
        viewer.add(np.transpose(inputs['image_input'][0, :, :, :, 0]),
                   name='Image')
        viewer.add(np.transpose(inputs['mask_input'][0, :, :, :, 0]),
                   name='Mask Input',
                   shader=get_color_shader(2))
        viewer.add(np.transpose(targets['mask_output'][0, :, :, :, 0]),
                   name='Mask Target',
                   shader=get_color_shader(0))
        output = ffn.predict(inputs)
        viewer.add(np.transpose(output[0, :, :, :, 0]),
                   name='Mask Output',
                   shader=get_color_shader(1))
        print(viewer)
    plot_history(history)
    return history

if __name__ == "__main__":
    main()
