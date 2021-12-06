import os
import unittest

from tools import Config


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")

    def test_split_data(self):
        self.assertEqual(
            len(self.config.dataset),
            len(self.config.train_ds) + len(self.config.validation_ds),
        )

    def test_get_filepaths(self):
        for i in range(self.config.num_validation):
            fp = self.config.validation_ds[i]
            self.assertTrue(os.path.isfile(fp))

    def test_get_fp_out_of_range(self):
        with self.assertRaises(IndexError):
            self.config.dataset[self.config.num_labels]
