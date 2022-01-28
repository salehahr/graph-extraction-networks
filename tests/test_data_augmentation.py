import unittest

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from tools import Config, GraphExtractionDG, NodeExtractionDG, TestType
from tools.plots import plot_augmented, plot_training_sample


class TestDataAugmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")

        cls.img_datagen = cls._init_old_data_augmenter()
        cls.aug_seed = 1

        cls.network_num, cls.training_data = cls._gen_training_data()

    @classmethod
    def _init_old_data_augmenter(cls):
        data_gen_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
        )

        return ImageDataGenerator(**data_gen_args)

    @classmethod
    def _gen_training_data(cls):
        """Generates training data based on network ID."""
        network_num = 2
        training_data = NodeExtractionDG(
            cls.config,
            cls.config.network.node_extraction,
            TestType.TRAINING,
            augmented=False,
        )

        return network_num, training_data

    def _get_samples(self):
        """Get first output of the first training batch."""
        x_batch, y_batch = self._get_first_batch()

        # expand dimensions for inputting to ImageDataGenerator.flow()
        x_first = np.expand_dims(x_batch[0].numpy(), axis=0)

        node_pos_first = np.expand_dims(y_batch[0][0].numpy(), axis=0)
        degrees_first = np.expand_dims(y_batch[1][0].numpy(), axis=0)
        node_types_first = np.expand_dims(y_batch[2][0].numpy(), axis=0)

        return x_first, node_pos_first, degrees_first, node_types_first

    def _get_first_batch(self):
        """Gets first training batch."""
        plot_training_sample(self.training_data, network=self.network_num, rows=1)

        # gets the first batch (>= 1 images in batch)
        return self.training_data[0]

    def test_old_data_aug(self):
        """
        For testing the method previously used for data augmentation.
        """
        skel_img, node_pos, degrees, node_types = self._get_samples()

        skel_img_aug = self._aug_imgs(skel_img)
        graph_aug = {
            "node_pos": self._aug_imgs(node_pos),
            "degrees": self._aug_imgs(degrees),
            "node_types": self._aug_imgs(node_types),
        }

        plot_augmented(skel_img_aug, graph_aug)

    def _aug_imgs(self, sample):
        img_iter = self.img_datagen.flow(sample, batch_size=1, seed=self.aug_seed)
        return [img_iter.next() for _ in range(4)]

    def test_data_aug(self):
        plot_training_sample(self.training_data, network=self.network_num, rows=3)

        self.training_data.augmented = True

        plot_training_sample(self.training_data, network=self.network_num, rows=3)


class TestThirdNetworkDataAugmentation(TestDataAugmentation):
    @classmethod
    def _gen_training_data(cls):
        """Generates training data based on network ID."""
        network_num = 3
        training_data = GraphExtractionDG(
            cls.config,
            cls.config.network.graph_extraction,
            TestType.TRAINING,
            augmented=False,
        )

        return network_num, training_data

    def _get_samples(self):
        """Get first output of the first training batch."""
        x_batch, y_batch = self._get_first_batch()
        b_skel_img, b_node_pos, b_degrees = x_batch

        # expand dimensions for inputting to ImageDataGenerator.flow()
        skel_img_first = np.expand_dims(b_skel_img[0].numpy(), axis=0)
        node_pos_first = np.expand_dims(b_node_pos[0].numpy(), axis=0)
        degrees_first = np.expand_dims(b_degrees[0].numpy(), axis=0)
        adj_matr_first = np.expand_dims(y_batch[0].numpy(), axis=0)

        return skel_img_first, node_pos_first, degrees_first, adj_matr_first

    def test_old_data_aug(self):
        """
        For testing the method previously used for data augmentation.
        """
        skel_img, node_pos, degrees, adj_matr = self._get_samples()

        skel_img_aug = self._aug_imgs(skel_img)
        graph_aug = {
            "node_pos": self._aug_imgs(node_pos),
            "degrees": self._aug_imgs(degrees),
            "adj_matr": self._aug_imgs(adj_matr),
        }

        plot_augmented(skel_img_aug, graph_aug)
