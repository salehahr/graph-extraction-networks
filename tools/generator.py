import numpy as np

from tensorflow.keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array

from tools.PolyGraph import PolyGraph
from tools.data import get_next_filepaths_from_ds
from tools.image import generate_outputs

from enum import Enum, unique


@unique
class TestType(Enum):
    TRAINING = 1
    VALIDATION = 2


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, config, test_type: TestType,
                 to_fit=True):
        # ids
        self.num_data = config.num_train if test_type is TestType.TRAINING else \
            config.num_validation

        # input
        self.ds = config.train_ds if test_type is TestType.TRAINING else \
            config.validation_ds

        # dimensions
        self.batch_size = config.batch_size
        self.img_dims = config.img_dims
        self.n_channels = config.img_channels
        self.n_classes = config.output_channels

        self.to_fit = to_fit

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(self.num_data / self.batch_size))

    def __getitem__(self, i):
        """
        Returns the i-th batch
        :param i: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # assert i < self.__len__()

        filepaths = [get_next_filepaths_from_ds(self.ds) for _ in range(self.batch_size)]
        skel_fps, graph_fps = zip(*filepaths)

        skel_imgs = self._generate_x_tensor(skel_fps)

        if self.to_fit:
            node_attributes = self._generate_y_tensor(graph_fps)
            return skel_imgs, node_attributes
        else:
            return skel_imgs

    def _generate_x_tensor(self, skel_fps: list):
        x_tensor = np.empty((self.batch_size, *self.img_dims, self.n_channels))

        for i, fp in enumerate(skel_fps):
            x_tensor[i, :, :, 0] = img_to_array(load_img(fp, grayscale=True), dtype=np.float32).squeeze()

        return x_tensor / np.float32(255)

    def _generate_y_tensor(self, graph_fps):
        y_node_pos = np.empty((self.batch_size, *self.img_dims, 1), dtype=np.uint8)
        y_degrees = np.empty((self.batch_size, *self.img_dims, 1), dtype=np.uint8)
        y_node_types = np.empty((self.batch_size, *self.img_dims, 1), dtype=np.uint8)

        for i, fp in enumerate(graph_fps):
            graph = PolyGraph.load(fp)
            output_matrices = generate_outputs(graph, self.img_dims[0])

            y_node_pos[i, :, :, 0] = output_matrices['node_pos'].squeeze()
            y_degrees[i, :, :, 0] = output_matrices['degrees'].squeeze()
            y_node_types[i, :, :, 0] = output_matrices['node_types'].squeeze()

        return y_node_pos, y_degrees, y_node_types
