from abc import ABC
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from tools.data import fp_to_adj_matr, fp_to_grayscale_img, fp_to_node_attributes

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

        # dimensions
        self.num_data = num_data
        self.ds = ds.batch(config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
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

        return skel_imgs, node_pos, degrees, node_types, adj_matr

    def _get_batch_fps(self, i: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Returns filepaths of the batch (skeletonised images and graphs)."""
        skel_fps = self.ds.skip(i).take(1).unbatch()

        def skel_to_graph(skel_fp):
            graph_fp = tf.strings.regex_replace(skel_fp, "skeleton", "graphs")
            graph_fp = tf.strings.regex_replace(graph_fp, "\.png", ".json")
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

        def to_adj_matr(fp: str) -> np.ndarray:
            return tf.numpy_function(func=fp_to_adj_matr, inp=[fp], Tout=tf.uint8)

        node_attrs = graph_fps.map(to_node_attributes).unbatch()
        adj_matr = graph_fps.map(to_adj_matr)

        node_pos = node_attrs.window(1, shift=3).flat_map(lambda x: x)
        degrees = node_attrs.skip(1).window(1, shift=3).flat_map(lambda x: x)
        node_types = node_attrs.skip(2).window(1, shift=3).flat_map(lambda x: x)

        return node_pos, degrees, node_types, adj_matr

    def _augment(self, seed: int, data: List[tf.data.Dataset]):
        def augment(x):
            return self._augment_tensor(x, seed=seed)

        return [d.map(augment) for d in data] if self.augmented else data

    @staticmethod
    def _augment_tensor(x: tf.Tensor, seed: int) -> tf.Tensor:
        x = tf.image.random_flip_left_right(x, seed=seed)
        x = tf.image.random_flip_up_down(x, seed=seed)
        return x

    def _rebatch(self, data: List[tf.data.Dataset]) -> List[tf.Tensor]:
        def rebatch(x):
            return (
                x.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
                .take(1)
                .get_single_element()
            )

        return [rebatch(d) for d in data]


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
        input_data = [skel_imgs, node_pos, degrees, node_types]
        input_data = self._augment(i, input_data)

        # rebatch
        skel_imgs, node_pos, degrees, node_types = self._rebatch(input_data)
        node_attrs = (node_pos, degrees, node_types)

        return skel_imgs, node_attrs


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

        # augment
        data = [skel_imgs, node_pos, degrees, adj_matrs]
        data = self._augment(i, data)

        # rebatch
        data = [*data[:3], data[-1].map(lambda x: tf.RaggedTensor.from_tensor(x))]
        skel_imgs, node_pos, degrees, adj_matrs = self._rebatch(data)

        return (skel_imgs, node_pos, degrees), adj_matrs
