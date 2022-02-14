import random
from abc import ABC
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf

from tools.adj_matr import transform_adj_matr
from tools.colours import rgb_red
from tools.data import (
    fp_to_adj_matr,
    fp_to_grayscale_img,
    fp_to_node_attributes,
    get_data_at_xy,
    rebatch,
    sorted_pos_list_from_image,
)
from tools.image import gen_pos_indices_img

from .TestType import TestType


def to_skel_img(fp):
    return fp_to_grayscale_img(fp)


class DataGenerator(tf.keras.utils.Sequence, ABC):
    """Generates batches of training/validation/test data."""

    def __init__(self, config, network, test_type: TestType, augmented: bool = True):
        # dataset settings
        self.test_type = test_type
        if test_type == TestType.TRAINING:
            num_data = config.num_train
            ds = config.training_ds
        elif test_type == TestType.VALIDATION:
            num_data = config.num_validation
            ds = config.validation_ds
        elif test_type == TestType.TESTING:
            num_data = config.num_test
            ds = config.test_ds
        else:
            raise Exception

        # shuffle dataset before batching
        ds = ds.shuffle(num_data, reshuffle_each_iteration=False)
        self.ds = ds.batch(config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)

        # dimensions
        self.num_data = num_data
        self.batch_size = config.batch_size
        self.img_dims = config.img_dims
        self.input_channels = network.input_channels
        self.output_channels = network.output_channels

        # data_augmentation
        self.augmented = augmented

        # shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        i.e. number of steps per epoch."""
        return int(np.floor(self.num_data / self.batch_size))

    def on_epoch_end(self):
        self.ds = self.ds.shuffle(self.num_data, reshuffle_each_iteration=False)

    def _get_data(self, i: int):
        skel_fps, graph_fps = self._get_batch_fps(i)

        skel_imgs = skel_fps.map(to_skel_img, num_parallel_calls=tf.data.AUTOTUNE)
        node_pos, degrees, node_types, adj_matr = self._get_graph_data(graph_fps)

        def set_shape(img: tf.Tensor) -> tf.Tensor:
            img.set_shape([*self.img_dims, 1])
            return img

        skel_imgs = skel_imgs.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)
        node_pos = node_pos.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)
        degrees = degrees.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)
        node_types = node_types.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)

        return skel_imgs, node_pos, degrees, node_types, adj_matr

    def _get_batch_fps(self, i: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Returns filepaths of the batch (skeletonised images and graphs)."""
        skel_fps = self.ds.skip(i).take(1).unbatch()

        def skel_to_graph(skel_fp):
            graph_fp = tf.strings.regex_replace(skel_fp, "skeleton", "graphs")
            graph_fp = tf.strings.regex_replace(graph_fp, r"\.png", ".json")
            return graph_fp

        graph_fps = skel_fps.map(skel_to_graph, num_parallel_calls=tf.data.AUTOTUNE)

        return skel_fps, graph_fps

    def _get_graph_data(
        self, graph_fps: tf.data.Dataset
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Generates node attribute tensors from graph.
        :param graph_fps: filepaths to the graph objects
        :return: node attributes and adjacency vector
        """

        def to_node_attributes(fp):
            return tf.numpy_function(
                func=fp_to_node_attributes, inp=[fp, self.img_dims[0]], Tout=tf.uint8
            )

        def to_adj_matr(fp: str):
            return tf.numpy_function(func=fp_to_adj_matr, inp=[fp], Tout=tf.uint8)

        node_attrs = graph_fps.map(to_node_attributes).unbatch()
        adj_matr = graph_fps.map(to_adj_matr).map(lambda x: tf.cast(x, tf.int32))

        node_pos = node_attrs.window(1, shift=3).flat_map(lambda x: x)
        degrees = node_attrs.skip(1).window(1, shift=3).flat_map(lambda x: x)
        node_types = node_attrs.skip(2).window(1, shift=3).flat_map(lambda x: x)

        return node_pos, degrees, node_types, adj_matr

    def _augment(self, data: List[tf.data.Dataset]):
        seed1 = random.randint(0, 100)
        seed2 = random.randint(0, 100)

        def augment(x):
            return self._flip_tensor(x, seeds=(seed1, seed2))

        return [d.map(augment, deterministic=True) for d in data]

    @staticmethod
    def _flip_tensor(x: tf.Tensor, seeds: tuple) -> tf.Tensor:
        x = tf.image.random_flip_left_right(x, seed=seeds[0])
        x = tf.image.random_flip_up_down(x, seed=seeds[1])
        return x

    def _rebatch(self, data: List[tf.data.Dataset]) -> List[tf.Tensor]:
        return [rebatch(d, self.batch_size) for d in data]


class NodeExtractionDG(DataGenerator):
    def __getitem__(
        self, i: int
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        """
        Returns the i-th batch.
        :param i: batch index
        :return: skeletonised images and node attributes of the images in the batch
        """
        skel_imgs, node_pos, degrees, node_types, _ = self._get_data(i)

        # augment
        data = [skel_imgs, node_pos, degrees, node_types]
        data = self._augment(data) if self.augmented else data

        # rebatch
        skel_imgs, node_pos, degrees, node_types = self._rebatch(data)

        return skel_imgs, (node_pos, degrees, node_types)


class GraphExtractionDG(DataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_nodes: int = 300
        self.max_adj_dim: int = int(self.max_nodes * (self.max_nodes - 1) / 2)

    def __getitem__(
        self, i: int
    ) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
        """Returns the i-th batch."""
        skel_imgs, node_pos, degrees, _, adj_matrs = self._get_data(i)
        pos_idx_img = self._to_pos_indices_img(node_pos)

        # augment
        data = [skel_imgs, node_pos, degrees, pos_idx_img]
        data = self._augment(data) if self.augmented else data

        pos_idx_aug_img = data[-1]
        adj_matrs = self._augment_adj_matr(adj_matrs, pos_idx_aug_img)

        # rebatch
        data = [*data[:3], adj_matrs]
        skel_imgs, node_pos, degrees, adj_matrs = self._rebatch(data)

        return (skel_imgs, node_pos, degrees), adj_matrs

    def _to_pos_indices_img(self, node_pos: tf.data.Dataset) -> tf.data.Dataset:
        """Generates an image dataset containing the integer indices
        of the node positions, at the corresponding (x,y) coordinates."""

        # get pos list from node_pos image
        pos_list = node_pos.map(
            sorted_pos_list_from_image, num_parallel_calls=tf.data.AUTOTUNE
        )

        # generate a range of values [0, n_nodes) = original node indices
        def gen_idx(pos: tf.Tensor):
            n_nodes = tf.shape(pos)[0]
            return tf.range(start=0, limit=n_nodes)

        idx = pos_list.map(gen_idx, num_parallel_calls=tf.data.AUTOTUNE)

        def to_img(i, xy):
            return tf.numpy_function(
                func=gen_pos_indices_img, inp=[i, xy, self.img_dims[0]], Tout=tf.uint32
            )

        return tf.data.Dataset.zip((idx, pos_list)).map(
            to_img, num_parallel_calls=tf.data.AUTOTUNE
        )

    @staticmethod
    def _augment_adj_matr(
        adj_matr: tf.data.Dataset, pos_idx_aug_img: tf.data.Dataset
    ) -> tf.data.Dataset:
        """Transforms adjacency matrix based on new node positions
        and returns it in RaggedTensor format."""

        def get_idx(img):
            return tf.numpy_function(func=get_data_at_xy, inp=[img], Tout=tf.uint32)

        aug_sort_indices = pos_idx_aug_img.map(get_idx)

        A_pos = tf.data.Dataset.zip((adj_matr, aug_sort_indices))
        return A_pos.map(transform_adj_matr, num_parallel_calls=tf.data.AUTOTUNE)


class EdgeExtractionDG(tf.keras.utils.Sequence):
    def __init__(
        self,
        config,
        network,
        test_type: TestType,
        skel_img: tf.Tensor,
        node_pos: tf.Tensor,
        degrees: tf.Tensor,
        adj_matr: tf.RaggedTensor,
        with_path: bool,
    ):
        self.test_type = test_type
        self.with_path = with_path

        self.skel_img = tf.squeeze(skel_img)
        self.node_pos = tf.squeeze(node_pos)
        self.degrees = tf.squeeze(degrees)
        self.adj_matr = adj_matr[0]

        # derived data
        self.skel_img_rgb = tf.stack(
            [self.skel_img, self.skel_img, self.skel_img], axis=-1
        )
        self.pos_list = sorted_pos_list_from_image(self.node_pos)

        n = tf.shape(self.pos_list)[0]
        self.num_nodes = n
        self.max_combinations = tf.cast(n * (n - 1) / 2, tf.int64)

        # dimensions
        self.batch_size = config.node_pairs_in_batch
        self.img_dims = config.img_dims
        self.input_channels = network.input_channels
        self.output_channels = network.output_channels

        # shuffle before batching
        all_combos = self._get_all_combinations()
        all_combos = all_combos.shuffle(
            self.max_combinations, reshuffle_each_iteration=False
        )
        self.all_combos = all_combos.batch(self.batch_size)

        # shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        i.e. number of steps per epoch."""
        return int(np.floor(self.max_combinations / self.batch_size))

    def __getitem__(
        self, i: int
    ) -> Tuple[tf.Tensor, Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]]:
        batch_combo = self.all_combos.skip(i).take(1).unbatch()

        x = self._get_combo_img(batch_combo)
        y = self._get_labels(batch_combo)

        return x, y

    def get_combo(self, i):
        return self.all_combos.skip(i).take(1).get_single_element()

    def _get_combo_img(self, batch_combo: tf.Tensor) -> tf.Tensor:
        def to_combo_img(pair: tf.Tensor):
            rc1, rc2 = self._to_coords(pair)
            img = tf.Variable(self.skel_img_rgb)

            img[rc1[0], rc1[1], :].assign(rgb_red)
            img[rc2[0], rc2[1], :].assign(rgb_red)

            return img

        imgs = batch_combo.map(to_combo_img, num_parallel_calls=tf.data.AUTOTUNE)
        return rebatch(imgs, self.batch_size)

    def _get_labels(
        self, batch_combo: tf.data.Dataset
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        def get_adjacency(pair: tf.Tensor) -> tf.Tensor:
            n1, n2 = pair[0], pair[1]
            return self.adj_matr[n1, n2]

        def to_path(adjacency: tf.Tensor, pair: tf.Tensor):
            rc1, rc2 = self._to_coords(pair)
            row_indices = tf.sort([rc1[0], rc2[0]])
            col_indices = tf.sort([rc1[1], rc2[1]])

            img_section = self.skel_img[
                row_indices[0] : row_indices[1] + 1,
                col_indices[0] : col_indices[1] + 1,
            ]

            return tf.math.multiply(
                tf.cast(adjacency, tf.float32),
                tf.RaggedTensor.from_tensor(img_section),
            )

        adj = batch_combo.map(get_adjacency, num_parallel_calls=tf.data.AUTOTUNE)
        path = None

        if self.with_path:
            path = tf.data.Dataset.zip((adj, batch_combo)).map(
                to_path, num_parallel_calls=tf.data.AUTOTUNE
            )
            path = rebatch(path, self.batch_size)

        adj = tf.reshape(rebatch(adj, self.batch_size), [self.batch_size, 1])

        return adj if not self.with_path else (adj, path)

    def _to_coords(self, pair: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        xy1 = self.pos_list[pair[0], :]
        xy2 = self.pos_list[pair[1], :]

        rc1 = tf.reverse(xy1, axis=[0])
        rc2 = tf.reverse(xy2, axis=[0])

        return rc1, rc2

    def _get_all_combinations(self) -> tf.data.Dataset:
        n = self.num_nodes
        indices = tf.range(n)

        y, x = tf.meshgrid(indices, indices)
        x = tf.expand_dims(x, 2)
        y = tf.expand_dims(y, 2)
        z = tf.concat([x, y], axis=2)

        all_combos = tf.constant([[0, 0]])  # placeholder
        for x in range(n - 1):
            # goes from 0 to n - 2
            aa = z[x + 1 :, x, :]
            all_combos = tf.concat([all_combos, aa], axis=0)
        all_combos = all_combos[1:, :]

        return tf.data.Dataset.from_tensor_slices(all_combos)
