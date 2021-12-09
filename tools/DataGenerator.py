from typing import List, Tuple, Union

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import Sequence

from tools.data import ds_to_list
from tools.image import generate_outputs
from tools.PolyGraph import PolyGraph

from .TestType import TestType


class DataGenerator(Sequence):
    """Generates training/validation data."""

    def __init__(self, config, test_type: TestType, augmented: bool = True):
        # dataset settings
        self.test_type = test_type
        if test_type == TestType.TRAINING:
            self.num_data = config.num_train
            self.ds = config.training_ds
        elif test_type == TestType.VALIDATION:
            self.num_data = config.num_validation
            self.ds = config.validation_ds
        else:
            raise Exception

        # dimensions
        self.batch_size = config.batch_size
        self.img_dims = config.img_dims
        self.input_channels = config.input_channels
        self.output_channels = config.output_channels

        # data_augmentation
        self.augmentation_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
        )
        self.augmented = augmented
        self.augmenter = (
            ImageDataGenerator(**self.augmentation_args) if augmented else None
        )

        # shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        i.e. number of steps per epoch."""
        return int(np.floor(self.num_data / self.batch_size))

    def on_epoch_end(self):
        self.ds = self.ds.shuffle(self.num_data, reshuffle_each_iteration=False)

    def __getitem__(
        self, i: int
    ) -> Union[
        np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ]:
        """
        Returns the i-th batch.
        :param i: batch index
        :return: X and Y when fitting. X only when predicting
        """
        only_input = self.test_type is TestType.VALIDATION
        skel_imgs, node_attributes = self.get_batch_data(i, only_input=only_input)

        if only_input:
            return skel_imgs
        else:
            return skel_imgs, node_attributes

    def get_batch_data(self, b: int, only_input: bool = False):
        batch_fps = self.ds.skip(b * self.batch_size).take(self.batch_size)

        skel_fps = ds_to_list(batch_fps)
        graph_fps = [
            fp.replace("skeleton", "graphs").replace(".png", ".json") for fp in skel_fps
        ]

        skel_imgs = self._generate_x_tensor(skel_fps, b)
        node_attributes = self._generate_y_tensor(graph_fps, b)

        if only_input:
            return skel_imgs, None
        else:
            return skel_imgs, node_attributes

    def _generate_x_tensor(self, skel_fps: List[str], seed: int) -> np.ndarray:
        """
        Generates normalised tensors of the skeletonised images.
        :param skel_fps: filepaths to the skeletonised images
        :return: skeletonised image tensor, normalised
        """
        x_tensor = np.empty((self.batch_size, *self.img_dims, self.input_channels))

        for i, fp in enumerate(skel_fps):
            x = img_to_array(load_img(fp, grayscale=True), dtype=np.float32)

            if self.augmented:
                x = self._augment_tensor(x, seed)

            x_tensor[i, :, :, 0] = x.squeeze()

        x_normalised = x_tensor / np.float32(255)

        return x_normalised.astype(np.float32)

    def _generate_y_tensor(
        self, graph_fps: List[str], seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates node attribute tensors from graph.
        :param graph_fps: filepaths to the graph objects
        :return: node position, node degree and node type tensors
        """
        y_node_pos = np.empty((self.batch_size, *self.img_dims, 1), dtype=np.uint8)
        y_degrees = np.empty((self.batch_size, *self.img_dims, 1), dtype=np.uint8)
        y_node_types = np.empty((self.batch_size, *self.img_dims, 1), dtype=np.uint8)

        for i, fp in enumerate(graph_fps):
            graph = PolyGraph.load(fp)
            output_matrices = generate_outputs(graph, self.img_dims[0])

            node_pos = output_matrices["node_pos"]
            degrees = self._cap_degrees(output_matrices["degrees"])
            node_types = output_matrices["node_types"]

            if self.augmented:
                node_pos = self._augment_tensor(node_pos, seed)
                degrees = self._augment_tensor(degrees, seed)
                node_types = self._augment_tensor(node_types, seed)

            y_node_pos[i, :, :, 0] = node_pos.squeeze()
            y_degrees[i, :, :, 0] = degrees.squeeze()
            y_node_types[i, :, :, 0] = node_types.squeeze()

        return y_node_pos, y_degrees, y_node_types

    def _augment_tensor(self, x: np.ndarray, seed: int) -> np.ndarray:
        x = np.expand_dims(x, axis=0)
        return self.augmenter.flow(x, batch_size=1, seed=seed)[0]

    @staticmethod
    def _cap_degrees(degrees):
        """Cap values at 4."""
        cap_value = 4
        degrees[degrees > cap_value] = cap_value
        return degrees
