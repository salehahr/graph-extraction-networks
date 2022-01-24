import unittest

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from tools import Config, GraphExtractionDG, NodeExtractionDG, TestType
from tools.plots import plot_augmented, plot_training_sample


class TestDataAugmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        network = cls.config.network.node_extraction
        cls.network_num = 2

        cls.filepaths = cls.config.dataset
        cls.training_data = NodeExtractionDG(
            cls.config, network, TestType.TRAINING, augmented=False
        )

        cls.img_datagen = cls._init_old_data_augmenter()

    @classmethod
    def _init_old_data_augmenter(cls):
        data_gen_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
        )

        return ImageDataGenerator(**data_gen_args)

    def _get_samples(self):
        """Get first output of the first training batch."""
        plot_training_sample(self.training_data, network=2, rows=1)

        # gets the first batch (>= 1 images in batch)
        # therefore the need for secondary indexing in x and the node attributes in y)
        x_batch, y_batch = self.training_data[0]

        # expand dimensions for inputting to ImageDataGenerator.flow()
        x_first = np.expand_dims(x_batch[0].numpy(), axis=0)

        node_pos_first = np.expand_dims(y_batch[0][0].numpy(), axis=0)
        degrees_first = np.expand_dims(y_batch[1][0].numpy(), axis=0)
        node_types_first = np.expand_dims(y_batch[2][0].numpy(), axis=0)

        return x_first, node_pos_first, degrees_first, node_types_first

    @unittest.skip("Now irrelevant.")
    def test_old_data_aug(self):
        """
        For testing the method previously used for data augmentation.
        (deprecated after updating DataGenerator to use more tf.data.Dataset
        functionality)
        """
        x, node_pos, degrees, node_types = self._get_samples()

        seed = 1
        x_iter = self.img_datagen.flow(x, batch_size=1, seed=seed)
        node_pos_iter = self.img_datagen.flow(node_pos, batch_size=1, seed=seed)
        degrees_iter = self.img_datagen.flow(degrees, batch_size=1, seed=seed)
        node_types_iter = self.img_datagen.flow(node_types, batch_size=1, seed=seed)

        y_iters = {
            "node_pos": node_pos_iter,
            "degrees": degrees_iter,
            "node_types": node_types_iter,
        }

        plot_augmented(x_iter, y_iters)

    def test_data_aug(self):
        plot_training_sample(self.training_data, network=self.network_num, rows=1)

        self.training_data.augmented = True

        plot_training_sample(self.training_data, network=self.network_num, rows=1)


class TestThirdNetworkDataAugmentation(TestDataAugmentation):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        network = cls.config.network.graph_extraction
        cls.network_num = 3

        cls.filepaths = cls.config.dataset
        cls.training_data = GraphExtractionDG(
            cls.config, network, TestType.TRAINING, augmented=False
        )
