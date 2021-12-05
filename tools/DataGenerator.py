from typing import List, Tuple, Union

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import Sequence

from tools.data import ds_to_list
from tools.image import generate_outputs
from tools.PolyGraph import PolyGraph

from .TestType import TestType


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, config, test_type: TestType):
        # dataset settings
        self.test_type = test_type
        if test_type == TestType.TRAINING:
            self.num_data = config.num_train
            self.ds = ds_to_list(config.dataset.take(config.num_train))
        elif test_type == TestType.VALIDATION:
            self.num_data = config.num_validation
            self.ds = ds_to_list(config.dataset.skip(config.num_train))
        else:
            raise Exception

        # ids associated with files in dataset, immutable
        self.data_ids = np.arange(self.num_data)
        self.data_ids.setflags(write=False)

        # ids of the files in the training/validation procedure
        # reshuffles every epoch
        self.test_ids = np.arange(self.num_data)

        # dimensions
        self.batch_size = config.batch_size
        self.img_dims = config.img_dims
        self.input_channels = config.input_channels
        self.output_channels = config.output_channels

        # shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        i.e. number of steps per epoch."""
        return int(np.floor(self.num_data / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.test_ids)

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
        assert i < len(self)

        batch_ids = np.arange(self.batch_size * i, self.batch_size * (i + 1))
        data_ids_for_batch = [self.data_ids[b] for b in batch_ids]

        skel_fps = [self.ds[fp] for fp in data_ids_for_batch]
        graph_fps = [
            fp.replace("skeleton", "graphs").replace(".png", ".json") for fp in skel_fps
        ]

        skel_imgs = self._generate_x_tensor(skel_fps)

        if self.test_type is TestType.TRAINING:
            node_attributes = self._generate_y_tensor(graph_fps)
            return skel_imgs, node_attributes
        else:
            return skel_imgs

    def _generate_x_tensor(self, skel_fps: List[str]) -> np.ndarray:
        """
        Generates normalised tensors of the skeletonised images.
        :param skel_fps: filepaths to the skeletonised images
        :return: skeletonised image tensor, normalised
        """
        x_tensor = np.empty((self.batch_size, *self.img_dims, self.input_channels))

        for i, fp in enumerate(skel_fps):
            x_tensor[i, :, :, 0] = img_to_array(
                load_img(fp, grayscale=True), dtype=np.float32
            ).squeeze()

        x_normalised = x_tensor / np.float32(255)

        return x_normalised.astype(np.float32)

    def _generate_y_tensor(
        self, graph_fps: List[str]
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

            y_node_pos[i, :, :, 0] = output_matrices["node_pos"].squeeze()
            y_degrees[i, :, :, 0] = output_matrices["degrees"].squeeze()
            y_node_types[i, :, :, 0] = output_matrices["node_types"].squeeze()

        return y_node_pos, y_degrees, y_node_types
