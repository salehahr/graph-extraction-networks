import unittest

import numpy as np

from tools import Config, DataGenerator, TestType
from tools.plots import plot_training_sample


class TestDataGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        cls.training_data = DataGenerator(cls.config, TestType.TRAINING)
        cls.validation_data = DataGenerator(cls.config, TestType.VALIDATION)

    def test_training_generator(self):
        self.assertEqual(len(self.training_data), 9)
        self._cycle_through_items(self.training_data)

    def test_validation_generator(self):
        self.assertEqual(len(self.validation_data), 1)
        self._cycle_through_items(self.validation_data)

    def _cycle_through_items(self, dataset):
        # one complete epoch: passes through all the data in the dataset
        for i in range(len(dataset)):
            self.assertIsNotNone(dataset[i])

    def test_plot_training_sample(self):
        plot_training_sample(self.training_data)

    def test_input_data(self):
        step_num = 0
        batch_id = 0
        x = self.validation_data[step_num][batch_id]

        is_normalised = np.max(x) <= 1

        self.assertTrue(is_normalised)
        self.assertEqual(x.shape, (256, 256, 1))
        self.assertEqual(x.dtype, np.float32)

    def test_output_data(self):
        step_num = 0
        batch_id = 0
        _, y = self.training_data[step_num]

        node_pos = y[0][batch_id]
        self.assertEqual(node_pos.shape, (256, 256, 1))
        self.assertEqual(node_pos.dtype, np.uint8)
        self.assertEqual(np.min(node_pos), 0)
        self.assertEqual(np.max(node_pos), 1)

        node_degrees = y[1][batch_id]
        self.assertEqual(node_degrees.shape, (256, 256, 1))
        self.assertEqual(node_degrees.dtype, np.uint8)
        self.assertEqual(np.min(node_pos), 0)
        self.assertLessEqual(np.max(node_pos), 5)

        node_types = y[2][batch_id]
        self.assertEqual(node_types.shape, (256, 256, 1))
        self.assertEqual(node_degrees.dtype, np.uint8)
        self.assertEqual(np.min(node_pos), 0)
        self.assertLessEqual(np.max(node_pos), 3)
