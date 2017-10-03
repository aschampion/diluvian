# -*- coding: utf-8 -*-


from __future__ import division

import itertools
import logging

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import six
from six.moves import queue
from tqdm import tqdm

from .config import CONFIG
from .octrees import OctreeVolume
from .postprocessing import Body
from .util import (
        get_color_shader,
        pad_dims,
        WrappedViewer,
        )


class Region(object):
    """A region (single seeded body) for flood filling.

    This object contains the necessary data to perform flood filling for a
    single body.

    Parameters
    ----------
    image : ndarray or diluvian.octrees.OctreeVolume
        Raw image data. If it is an octree, it is assumed all volumetric
        storage should also operate block-sparsely and there is no ground
        truth available.
    target : ndarray, optional
        Target mask probabilities (ground truth converted to network targets).
    seed_vox : ndarray, optional
        Coordinates of the seed voxel.
    mask : ndarray, optional
        Object prediction mask for output. Provided as an argument here in
        case resuming or extending an existing result.
    sparse_mask : bool, optional
        If true, force the predicted mask to be a block-sparse array instead
        of a dense ndarray. Note that if ``image`` is a
        ``diluvian.octrees.OctreeVolume``, the mask will be sparse regardless
        of this parameter.
    block_padding : str, optional
        Method to use to pad data when the network's input field of view
        extends outside the region bounds. This is passed to ``numpy.pad``.
        Defaults to ``None``, which indicates attempts to operate outside the
        region bounds are erroneous.

    Attributes
    ----------
    bias_against_merge : bool
        Whether to bias against merge by never overwriting mask probabilities
        less than 0.5 once they have been written.
    move_based_on_new_mask : bool
        Whether to generate moves based on the probabilities only in the newly
        predicted mask block (if true), or on the mask block once combined with
        the existing probability mask (if false).
    move_check_thickness : int
        Thickness in voxels to check around the move plane in each direction
        when determining which moves to queue. See ``get_moves`` method.
    """

    @staticmethod
    def from_subvolume(subvolume):
        if subvolume.label_mask is not None and np.issubdtype(subvolume.label_mask.dtype, np.bool):
            target = mask_to_output_target(subvolume.label_mask)
        else:
            target = subvolume.label_mask
        return Region(subvolume.image,
                      target=target,
                      seed_vox=subvolume.seed)

    @staticmethod
    def from_subvolume_generator(subvolumes):
        subvolumes = itertools.ifilter(lambda s: s.has_uniform_seed_margin(), subvolumes)
        return itertools.imap(Region.from_subvolume, subvolumes)

    def __init__(self, image, target=None, seed_vox=None, mask=None, sparse_mask=False, block_padding=None):
        self.block_padding = block_padding
        self.MOVE_DELTA = CONFIG.model.move_step
        self.queue = queue.PriorityQueue()
        self.visited = set()
        self.image = image
        self.bounds = np.array(image.shape, dtype=np.int64)
        if seed_vox is None:
            self.MOVE_GRID_OFFSET = np.array([0, 0, 0], dtype=np.int64)
        else:
            self.MOVE_GRID_OFFSET = np.mod(seed_vox, self.MOVE_DELTA).astype(np.int64)
        self.move_bounds = (
            np.ceil(np.true_divide((CONFIG.model.input_fov_shape - 1) // 2 - self.MOVE_GRID_OFFSET,
                                   self.MOVE_DELTA)).astype(np.int64),
            self.vox_to_pos(np.array(self.bounds) - 1 - (CONFIG.model.input_fov_shape - 1) // 2),
            )
        self.move_check_thickness = CONFIG.model.move_check_thickness
        if mask is None:
            if isinstance(self.image, OctreeVolume):
                self.mask = OctreeVolume(self.image.leaf_shape, (np.zeros(3), self.bounds), 'float32')
                self.mask[:] = np.NAN
            elif sparse_mask:
                self.mask = OctreeVolume(CONFIG.model.training_subv_shape, (np.zeros(3), self.bounds), 'float32')
                self.mask[:] = np.NAN
            else:
                self.mask = np.full(self.bounds, np.NAN, dtype=np.float32)
        else:
            self.mask = mask
        self.target = target

        self.bias_against_merge = False
        self.move_based_on_new_mask = False
        self.prioritize_proximity = CONFIG.model.move_priority == 'proximity'
        self.proximity = {}

        if seed_vox is None:
            seed_pos = np.floor_divide(self.move_bounds[0] + self.move_bounds[1], 2)
        else:
            seed_pos = self.vox_to_pos(seed_vox)
            assert self.pos_in_bounds(seed_pos), \
                'Seed position (%s) must be in region move bounds (%s, %s).' % \
                (seed_vox, self.move_bounds[0], self.move_bounds[1])
        self.seed_pos = seed_pos
        self.queue.put((None, seed_pos))
        self.proximity[tuple(seed_pos)] = 1
        self.seed_vox = self.pos_to_vox(seed_pos)
        if self.target is not None:
            self.target_offset = (self.bounds - self.target.shape) // 2
            assert np.isclose(self.target[tuple(self.seed_vox - self.target_offset)], CONFIG.model.v_true), \
                'Seed position should be in target body.'
        self.mask[tuple(self.seed_vox)] = CONFIG.model.v_true
        self.visited.add(tuple(self.seed_pos))

    def unfilled_copy(self):
        """Clone this region in an initial state without any filling.

        Returns
        -------
        Region
        """
        copy = Region(self.image, target=self.target, seed_vox=self.pos_to_vox(self.seed_pos))
        copy.bias_against_merge = self.bias_against_merge
        copy.move_based_on_new_mask = self.move_based_on_new_mask

        return copy

    def to_body(self):
        def threshold(a):
            return a >= CONFIG.model.t_final

        if isinstance(self.mask, OctreeVolume):
            hard_mask = self.mask.map_copy(np.bool, threshold, threshold)
        else:
            hard_mask = threshold(self.mask)

        return Body(hard_mask, self.pos_to_vox(self.seed_pos))

    def vox_to_pos(self, vox):
        return np.floor_divide(vox - self.MOVE_GRID_OFFSET, self.MOVE_DELTA).astype(np.int64)

    def pos_to_vox(self, pos):
        return (pos * self.MOVE_DELTA).astype(np.int64) + self.MOVE_GRID_OFFSET

    def pos_in_bounds(self, pos):
        if self.block_padding is None:
            return np.all(np.greater_equal(pos, self.move_bounds[0])) and \
                np.all(np.less_equal(pos, self.move_bounds[1]))
        else:
            return np.all(np.less(self.pos_to_vox(pos), self.bounds)) and np.all(pos >= 0)

    def get_block_bounds(self, vox, shape, offset=None):
        """Get the bounds of a block by center and shape, accounting padding.

        Returns the voxel bounds of a block specified by shape and center in
        the region, clamping the bounds to be in the volume but returning
        padding margins that extend outside the region bounds.

        Parameters
        ----------
        vox : ndarray
            Center of the block in voxel coordinates.
        shape : ndarray
            Shape of the block.
        offset : ndarray, optional
            If provided, offset of coordinates from the volume where these
            bounds where be used. This is needed if the volume has a margin
            (i.e., is smaller than the main region volume), such as the target
            volume for contracted output shapes.

        Returns
        -------
        block_min, block_max : ndarray
            Extents of the block in voxel coordinates clamped to the region
            bounds.
        padding_pre, padding_post : ndarray
            How much the block extends outside the region bounds.
        """
        if offset is None:
            offset = np.zeros(3, dtype=vox.dtype)
        margin = (shape - 1) // 2
        block_min = vox - margin
        block_max = vox + margin + 1
        padding_pre = np.maximum(0, -block_min)
        padding_post = np.maximum(0, block_max - self.bounds + offset + offset)

        block_min = np.maximum(0, block_min)
        block_max = np.minimum(block_max, self.bounds - offset - offset)

        return block_min, block_max, padding_pre, padding_post

    def get_moves(self, mask):
        """Given a mask block, get maximum probability in each move direction.

        Checks each of six planes comprising a centered cube half the shape
        of the provided block. For each of these planes, the maximum
        probability in the mask block is returned along with the move
        direction.

        Unlike the original implementation, this will check an n-voxel thick
        slab of voxels around each pane specified by this region's
        ``move_check_thickness`` property. This is useful for overcoming
        artifacts that may only affect a single plane that happens to align
        with the move grid.

        Parameters
        ----------
        mask : ndarray
            Block of mask probabilities, usually of the shape specified by
            the configured ``output_fov_shape``.

        Returns
        -------
        list of dict
            Each dict should include a ``move`` ndarray unit vector indicating
            the move direction and a ``v`` indicating the max probability
            in the move plane in that direction.
        """
        moves = []
        ctr = np.asarray(mask.shape) // 2
        for move in map(np.array, [(1, 0, 0), (-1, 0, 0),
                                   (0, 1, 0), (0, -1, 0),
                                   (0, 0, 1), (0, 0, -1)]):
            plane_min = ctr - (-2 * np.maximum(move, 0) + 1) * self.MOVE_DELTA \
                            - np.abs(move) * (self.move_check_thickness - 1)
            plane_max = ctr + (+2 * np.minimum(move, 0) + 1) * self.MOVE_DELTA \
                            + np.abs(move) * (self.move_check_thickness - 1) + 1
            moves.append({'move': move,
                          'v': mask[plane_min[0]:plane_max[0],
                                    plane_min[1]:plane_max[1],
                                    plane_min[2]:plane_max[2]].max()})
        return moves

    def check_move_neighborhood(self, mask):
        """Given a mask block, check if any central voxels meet move threshold.

        Checks whether a cube one move in each direction from the mask center
        contains any probabilities greater than the move threshold.

        Parameters
        ----------
        mask : ndarray
            Block of mask probabilities, usually of the shape specified by
            the configured ``output_fov_shape``.

        Returns
        -------
        bool
        """
        ctr = np.asarray(mask.shape) // 2
        neigh_min = ctr - self.MOVE_DELTA
        neigh_max = ctr + self.MOVE_DELTA + 1
        neighborhood = mask[map(slice, neigh_min, neigh_max)]
        return neighborhood.max() >= CONFIG.model.t_move

    def add_mask(self, mask_block, mask_pos):
        mask_vox = self.pos_to_vox(mask_pos)
        mask_min, mask_max, pad_pre, pad_post = self.get_block_bounds(mask_vox, np.asarray(mask_block.shape))

        if np.any(pad_pre) or np.any(pad_post):
            assert self.block_padding is not None, \
                'Position block extends out of region bounds, but padding is not enabled: {}'.format(mask_pos)
            end = [-x if x != 0 else None for x in pad_post]
            mask_block = mask_block[map(slice, pad_pre, end)]
        current_mask = self.mask[mask_min[0]:mask_max[0],
                                 mask_min[1]:mask_max[1],
                                 mask_min[2]:mask_max[2]]

        if self.bias_against_merge:
            update_mask = np.isnan(current_mask) | (current_mask > 0.5) | np.less(mask_block, current_mask)
            current_mask[update_mask] = mask_block[update_mask]
        else:
            current_mask[:] = mask_block

        self.mask[mask_min[0]:mask_max[0],
                  mask_min[1]:mask_max[1],
                  mask_min[2]:mask_max[2]] = current_mask

        if self.move_based_on_new_mask:
            move_check_block = mask_block
        else:
            move_check_block = current_mask
        pad_width = zip(list(pad_pre), list(pad_post))
        move_check_block = np.pad(move_check_block, pad_width, 'constant')

        new_moves = self.get_moves(move_check_block)
        if self.prioritize_proximity:
            proximity = self.proximity[tuple(mask_pos)] + 1
            del self.proximity[tuple(mask_pos)]
        for move in new_moves:
            new_pos = mask_pos + move['move']
            if not self.pos_in_bounds(new_pos):
                continue
            if tuple(new_pos) not in self.visited and move['v'] >= CONFIG.model.t_move:
                self.visited.add(tuple(new_pos))
                priority = -move['v']
                if self.prioritize_proximity:
                    self.proximity[tuple(new_pos)] = min(self.proximity.get(tuple(new_pos), proximity), proximity)
                    priority /= proximity
                self.queue.put((priority, tuple(new_pos)))

    def get_next_block(self):
        try:
            queued_move = self.queue.get_nowait()
        except queue.Empty:
            return None

        next_pos = np.asarray(queued_move[1])
        next_vox = self.pos_to_vox(next_pos)
        block_min, block_max, pad_pre, pad_post = self.get_block_bounds(next_vox, CONFIG.model.input_fov_shape)

        assert self.block_padding is not None or not (np.any(pad_pre) or np.any(pad_post)), \
            'Position block extends out of region bounds, but padding is not enabled: {}'.format(next_pos)

        mask_block = self.mask[block_min[0]:block_max[0],
                               block_min[1]:block_max[1],
                               block_min[2]:block_max[2]].copy()

        mask_block[np.isnan(mask_block)] = CONFIG.model.v_false

        # Check that there is still some t_move threshold mask near the move.
        if CONFIG.model.move_recheck and not (
           np.array_equal(next_pos, self.seed_pos) or self.check_move_neighborhood(mask_block)):
            logging.debug('Skipping move: no threshold mask in cube around voxel %s', np.array_str(next_vox))
            return self.get_next_block()

        image_block = self.image[block_min[0]:block_max[0],
                                 block_min[1]:block_max[1],
                                 block_min[2]:block_max[2]]

        if np.any(pad_pre) or np.any(pad_post):
            assert self.block_padding is not None, \
                'Position block extends out of region bounds, but padding is not enabled: {}'.format(next_pos)
            pad_width = zip(list(pad_pre), list(pad_post))
            image_block = np.pad(image_block, pad_width, self.block_padding)
            mask_block = np.pad(mask_block, pad_width, self.block_padding)

        if self.target is not None:
            block_min, block_max, pad_pre, pad_post = self.get_block_bounds(
                    next_vox - self.target_offset, CONFIG.model.output_fov_shape, self.target_offset)
            target_block = self.target[block_min[0]:block_max[0],
                                       block_min[1]:block_max[1],
                                       block_min[2]:block_max[2]]
            if np.any(pad_pre) or np.any(pad_post):
                pad_width = zip(list(pad_pre), list(pad_post))
                target_block = np.pad(target_block, pad_width, self.block_padding)
        else:
            target_block = None

        assert image_block.shape == tuple(CONFIG.model.input_fov_shape), \
            'Image wrong shape: {}'.format(image_block.shape)
        assert mask_block.shape == tuple(CONFIG.model.input_fov_shape), \
            'Mask wrong shape: {}'.format(mask_block.shape)
        return {'image': image_block,
                'mask': mask_block,
                'target': target_block,
                'position': next_pos}

    def remask(self):
        """Reset the mask based on the seeded connected component.
        """
        body = self.to_body()
        if not body.is_seed_in_mask():
            return False
        new_mask_bin, bounds = body.get_seeded_component(CONFIG.postprocessing.closing_shape)
        new_mask_bin = new_mask_bin.astype(np.bool)

        mask_block = self.mask[map(slice, bounds[0], bounds[1])]
        # Clip any values not in the seeded connected component so that they
        # cannot not generate moves when rechecking.
        mask_block[~new_mask_bin] = np.clip(mask_block[~new_mask_bin], None, 0.9 * CONFIG.model.t_move)

        self.mask[:] = np.NAN
        self.mask[map(slice, bounds[0], bounds[1])] = mask_block
        return True

    def fill(self, model, progress=False, move_batch_size=1, max_moves=None, stopping_callback=None,
             remask_interval=None):
        """Flood fill this region.

        Parameters
        ----------
        model : keras.models.Model
            Model to use for object prediction.
        progress : bool or int, optional
            Whether to display a progress bar. If an int, indicates the
            progress bar is nested and should appear at that level.
        move_batch_size : int, optional
            Number of moves to process in parallel. Note that in the algorithm
            as originally described this is 1, because otherwise moves'
            outputs may affect each other or the queue. Setting this higher
            can increase throughput.
        max_moves : int, optional
            Terminate filling after this many moves even if the queue is not
            empty.
        stopping_callback : function, optional
            Function periodical called that will terminate filling if it returns
            true.
        remask_interval : int, optional
            Frequency in moves to reset the mask to be based on the
            morphological seed connected component. This is useful to discourage
            long-running fills due to runaway merging. Only sensible when
            using move rechecking, proximity priority, and rejecting non-seeded
            connected components.

        Returns
        -------
        bool
            Whether filling was terminated early due to either exceedind the
            maximum number of moves or the stopping callback.
        """
        moves = 0
        last_check = 0
        last_remask = 0
        STOP_CHECK_INTERVAL = 100
        early_termination = False

        if progress:
            pbar = tqdm(desc='Move queue', position=progress)
        while not self.queue.empty():
            batch_block_data = [self.get_next_block() for _ in
                                itertools.takewhile(lambda _: not self.queue.empty(), range(move_batch_size))]
            batch_block_data = [b for b in batch_block_data if b is not None]
            batch_moves = len(batch_block_data)
            if batch_moves == 0:
                break
            moves += batch_moves
            if progress:
                pbar.total = moves + self.queue.qsize()
                pbar.set_description(str(self.seed_vox) + ' Move ' + str(batch_block_data[-1]['position']))
                pbar.update(batch_moves)

            if stopping_callback is not None and moves - last_check >= STOP_CHECK_INTERVAL:
                last_check = moves
                if stopping_callback(self):
                    early_termination = True
                    break

            image_input = np.concatenate([pad_dims(b['image']) for b in batch_block_data])
            mask_input = np.concatenate([pad_dims(b['mask']) for b in batch_block_data])

            output = model.predict_on_batch({'image_input': image_input,
                                             'mask_input': mask_input})

            for ind, block_data in enumerate(batch_block_data):
                self.add_mask(output[ind, :, :, :, 0], block_data['position'])

            if max_moves is not None and moves > max_moves:
                early_termination = True
                break

            if moves - last_remask >= remask_interval:
                if not self.remask():
                    early_termination = True
                    break
                last_remask = moves

        if progress:
            pbar.close()

        return early_termination

    def fill_animation(self, model, movie_filename, verbose=False):
        """Create an animated movie of the filling process for this region.

        .. note:: Deprecated
                  This method is not maintained so has not been updated to
                  reflect many changes. It is kept as a template but would
                  need rewriting to be functional.
        """
        dpi = 100
        fig = plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
        fig.patch.set_facecolor('black')
        axes = {
            'xy': fig.add_subplot(1, 3, 1),
            'xz': fig.add_subplot(1, 3, 2),
            'zy': fig.add_subplot(1, 3, 3),
        }

        def get_plane(arr, vox, plane):
            return {
                'xy': lambda a, v: np.transpose(a[:, :, v[2]]),
                'xz': lambda a, v: np.transpose(a[:, v[1], :]),
                'zy': lambda a, v: a[v[0], :, :],
            }[plane](arr, np.round(vox).astype(np.int64))

        def get_hv(vox, plane):
            # rel = np.divide(vox, self.bounds)
            rel = vox
            # rel = self.bounds - vox
            return {
                'xy': {'h': rel[1], 'v': rel[0]},
                'xz': {'h': rel[2], 'v': rel[0]},
                'zy': {'h': rel[1], 'v': rel[2]},
            }[plane]

        def get_aspect(plane):
            return {
                'xy': float(CONFIG.volume.resolution[1]) / float(CONFIG.volume.resolution[0]),
                'xz': float(CONFIG.volume.resolution[2]) / float(CONFIG.volume.resolution[0]),
                'zy': float(CONFIG.volume.resolution[1]) / float(CONFIG.volume.resolution[2]),
            }[plane]

        images = {
            'image': {},
            'mask': {},
        }
        lines = {
            'v': {},
            'h': {},
            'bl': {},
            'bt': {},
        }
        current_vox = self.pos_to_vox(self.seed_pos)
        margin = (CONFIG.model.input_fov_shape) // 2
        for plane, ax in six.iteritems(axes):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            image_data = get_plane(self.image, current_vox, plane)
            im = ax.imshow(image_data, cmap='gray')
            im.set_clim([0, 1])
            images['image'][plane] = im

            mask_data = get_plane(self.mask, current_vox, plane)
            im = ax.imshow(mask_data, cmap='jet', alpha=0.8)
            im.set_clim([0, 1])
            images['mask'][plane] = im

            aspect = get_aspect(plane)
            lines['h'][plane] = ax.axhline(y=get_hv(current_vox - margin, plane)['h'], color='w')
            lines['v'][plane] = ax.axvline(x=get_hv(current_vox + margin, plane)['v'], color='w')
            lines['bl'][plane] = ax.axvline(x=get_hv(current_vox - margin, plane)['v'], color='w')
            lines['bt'][plane] = ax.axhline(y=get_hv(current_vox + margin, plane)['h'], color='w')

            ax.set_aspect(aspect)

        plt.tight_layout()

        def update_fn(vox):
            if np.array_equal(np.round(vox).astype(np.int64), update_fn.next_pos_vox):
                if update_fn.block_data is not None:
                    image_input = pad_dims(update_fn.block_data['image'])
                    mask_input = pad_dims(update_fn.block_data['mask'])

                    output = model.predict({'image_input': image_input,
                                            'mask_input': mask_input})
                    self.add_mask(output[0, :, :, :, 0], update_fn.block_data['position'])

                if not self.queue.empty():
                    update_fn.moves += 1
                    if verbose:
                        update_fn.pbar.total = update_fn.moves + self.queue.qsize()
                        update_fn.pbar.update()
                    update_fn.block_data = self.get_next_block()

                    update_fn.next_pos_vox = self.pos_to_vox(update_fn.block_data['position'])
                    if not np.array_equal(np.round(vox).astype(np.int64), update_fn.next_pos_vox):
                        p = update_fn.next_pos_vox - vox
                        steps = np.linspace(0, 1, 16)
                        interp_vox = vox + np.outer(steps, p)
                        for row in interp_vox:
                            update_fn.vox_queue.put(row)
                    else:
                        update_fn.vox_queue.put(vox)

            for plane, im in six.iteritems(images['image']):
                image_data = get_plane(self.image, vox, plane)
                im.set_data(image_data)

            for plane, im in six.iteritems(images['mask']):
                image_data = get_plane(self.mask, vox, plane)
                masked_data = np.ma.masked_where(image_data < 0.5, image_data)
                im.set_data(masked_data)

            for plane in axes.iterkeys():
                lines['h'][plane].set_ydata(get_hv(vox - margin, plane)['h'])
                lines['v'][plane].set_xdata(get_hv(vox + margin, plane)['v'])
                lines['bl'][plane].set_xdata(get_hv(vox - margin, plane)['v'])
                lines['bt'][plane].set_ydata(get_hv(vox + margin, plane)['h'])

            return images['image'].values() + images['mask'].values() + \
                lines['h'].values() + lines['v'].values() + \
                lines['bl'].values() + lines['bt'].values()

        update_fn.moves = 0
        update_fn.block_data = None
        if verbose:
            update_fn.pbar = tqdm(desc='Move queue')

        update_fn.next_pos_vox = current_vox
        update_fn.vox_queue = queue.Queue()
        update_fn.vox_queue.put(current_vox)

        def vox_gen():
            last_vox = None
            while 1:
                if update_fn.vox_queue.empty():
                    return
                else:
                    last_vox = update_fn.vox_queue.get()
                    yield last_vox

        ani = animation.FuncAnimation(fig, update_fn, frames=vox_gen(), interval=16, repeat=False, save_count=60*60)
        writer = animation.writers['ffmpeg'](fps=60)

        ani.save(movie_filename, writer=writer, dpi=dpi, savefig_kwargs={'facecolor': 'black'})

        if verbose:
            update_fn.pbar.close()

        return ani

    def get_viewer(self, transpose=False):
        if transpose:
            viewer = WrappedViewer(voxel_size=list(CONFIG.volume.resolution),
                                   voxel_coordinates=self.pos_to_vox(self.seed_pos))
            viewer.add(np.transpose(self.image),
                       name='Image')
            if self.target is not None:
                viewer.add(np.transpose(self.target),
                           name='Mask Target',
                           shader=get_color_shader(0))
            viewer.add(np.transpose(self.mask),
                       name='Mask Output',
                       shader=get_color_shader(1))
        else:
            viewer = WrappedViewer(voxel_size=list(np.flipud(CONFIG.volume.resolution)),
                                   voxel_coordinates=np.flipud(self.pos_to_vox(self.seed_pos)))
            viewer.add(self.image,
                       name='Image')
            if self.target is not None:
                viewer.add(self.target,
                           name='Mask Target',
                           shader=get_color_shader(0))
            viewer.add(self.mask,
                       name='Mask Output',
                       shader=get_color_shader(1))
        return viewer


def mask_to_output_target(mask):
    target = np.full_like(mask, CONFIG.model.v_false, dtype=np.float32)
    target[mask] = CONFIG.model.v_true
    return target
