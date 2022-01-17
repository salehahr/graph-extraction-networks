import unittest

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from tools import Config, DataGenerator, TestType
from tools.plots import plot_augmented, plot_training_sample


class TestDataAugmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        network = cls.config.network.node_extraction

        cls.filepaths = cls.config.dataset
        cls.training_data = DataGenerator(
            cls.config, network, TestType.TRAINING, augmented=False
        )

        cls.img_datagen = cls._init_data_augmenter()

    @classmethod
    def _init_data_augmenter(cls):
        data_gen_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
        )

        return ImageDataGenerator(**data_gen_args)

    def _get_samples(self):
        """Get first output of the first training batch."""
        # gets the first batch (>= 1 images in batch
        # therefore the need for secondary indexing in x and the node attributes in y)
        x, y = self.training_data[0]

        # expand dimensions for inputting to ImageDataGenerator.flow()
        x_first = np.expand_dims(x[0], axis=0)
        plot_training_sample(self.training_data, rows=1)

        node_pos, degrees, node_types = y
        node_pos_first = np.expand_dims(node_pos[0], axis=0)
        degrees_first = np.expand_dims(degrees[0], axis=0)
        node_types_first = np.expand_dims(node_types[0], axis=0)

        return x_first, node_pos_first, degrees_first, node_types_first

    def test_data_aug(self):
        """Visual test to check whether the data augmentation
        transfers properly to the node attribute matrices."""
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
