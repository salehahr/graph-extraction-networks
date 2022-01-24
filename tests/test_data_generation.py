import unittest

import numpy as np

from tools import Config, GraphExtractionDG, NodeExtractionDG, TestType
from tools.data import ds_to_list
from tools.plots import plot_training_sample


class TestNodeExtractionDG(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        network = cls.config.network.node_extraction

        cls.training_data = NodeExtractionDG(cls.config, network, TestType.TRAINING)
        cls.validation_data = NodeExtractionDG(cls.config, network, TestType.VALIDATION)

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
        plot_training_sample(self.training_data, network=2)

    def test_input_data(self):
        step_num = 0
        batch_id = 0

        x_batch, _ = self.validation_data[step_num]
        x = x_batch[batch_id].numpy()

        is_normalised = np.max(x) <= 1

        self.assertTrue(is_normalised)
        self.assertEqual(x.shape, (256, 256, 1))
        self.assertEqual(x.dtype, np.float32)

    def test_output_data(self):
        step_num = 0
        batch_id = 0

        _, y_batch = self.training_data[step_num]

        node_pos = y_batch[0][batch_id].numpy()
        self.assertEqual(node_pos.shape, (256, 256, 1))
        self.assertEqual(node_pos.dtype, np.uint8)
        self.assertEqual(np.min(node_pos), 0)
        self.assertEqual(np.max(node_pos), 1)

        node_degrees = y_batch[1][batch_id].numpy()
        self.assertEqual(node_degrees.shape, (256, 256, 1))
        self.assertEqual(node_degrees.dtype, np.uint8)
        self.assertEqual(np.min(node_degrees), 0)
        self.assertLessEqual(np.max(node_degrees), 4)

        node_types = y_batch[2][batch_id].numpy()
        self.assertEqual(node_types.shape, (256, 256, 1))
        self.assertEqual(node_types.dtype, np.uint8)
        self.assertEqual(np.min(node_types), 0)
        self.assertLessEqual(np.max(node_types), 3)


class TestGraphExtractionDG(TestNodeExtractionDG):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        network = cls.config.network.graph_extraction

        cls.training_data = GraphExtractionDG(cls.config, network, TestType.TRAINING)
        cls.validation_data = GraphExtractionDG(
            cls.config, network, TestType.VALIDATION
        )

    def test_plot_training_sample(self):
        plot_training_sample(self.training_data, network=3)

    def test_input_data(self):
        step_num = 0
        batch_id = 0

        x_batch, _ = self.validation_data[step_num]

        skel_img = x_batch[0][batch_id].numpy()
        is_normalised = np.max(skel_img) <= 1
        self.assertTrue(is_normalised)
        self.assertEqual(skel_img.shape, (256, 256, 1))
        self.assertEqual(skel_img.dtype, np.float32)

        node_pos = x_batch[1][batch_id].numpy()
        self.assertEqual(node_pos.shape, (256, 256, 1))
        self.assertEqual(node_pos.dtype, np.uint8)
        self.assertEqual(np.min(node_pos), 0)
        self.assertEqual(np.max(node_pos), 1)

        node_degrees = x_batch[2][batch_id].numpy()
        self.assertEqual(node_degrees.shape, (256, 256, 1))
        self.assertEqual(node_degrees.dtype, np.uint8)
        self.assertEqual(np.min(node_degrees), 0)
        self.assertLessEqual(np.max(node_degrees), 4)

    def test_output_data(self):
        step_num = 0
        batch_id = 0

        _, y_batch = self.training_data[step_num]
        adj_matr = y_batch[batch_id].numpy()

        self.assertEqual(adj_matr.ndim, 3)
        self.assertEqual(adj_matr.dtype, np.uint8)
        self.assertEqual(np.min(adj_matr), 0)
        self.assertLessEqual(np.max(adj_matr), 1)


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        assert cls.config.use_small_dataset is True
        assert cls.config.max_files == 30

        cls.ds = cls.config.dataset
        cls.test_ds = cls.config.test_ds
        cls.list_of_files = ds_to_list(cls.ds) + ds_to_list(cls.test_ds)

        print(cls.list_of_files)

    def test_two_takes(self):
        """
        Test to ensure that 'taking' from the same dataset returns the same filepaths.
        (Sanity check for shuffled dataset)
        """
        ds1 = self.ds.take(self.config.num_validation)
        ds2 = self.ds.take(self.config.num_validation)
        ds3 = self.ds.skip(self.config.num_validation).take(self.config.num_validation)

        list_files1 = ds_to_list(ds1)
        list_files2 = ds_to_list(ds2)
        list_files3 = ds_to_list(ds3)

        print("1")
        for fp in list_files1:
            print(fp)
            self.assertTrue(fp in self.list_of_files)

        print("2")
        for fp in list_files2:
            print(fp)
            self.assertTrue(fp in self.list_of_files)
            self.assertTrue(fp in list_files1)

        print("3")
        for fp in list_files3:
            print(fp)
            self.assertTrue(fp in self.list_of_files)
            self.assertFalse(fp in list_files2)

    def test_train_and_val_ds(self):
        """
        Ensures that none of the train/val filepaths are found outside of the
        main dataset.
        """
        train_ds = self.config.training_ds
        val_ds = self.config.validation_ds

        for fp in ds_to_list(train_ds):
            self.assertTrue(fp in self.list_of_files)

        for fp in ds_to_list(val_ds):
            self.assertTrue(fp in self.list_of_files)

    def test_simulate_epoch(self):
        """
        Simulates running through the dataset for one epoch.
        """
        batch_size = 3
        steps_per_epoch = int(len(self.ds) / batch_size)

        batched_ds = self.ds.batch(batch_size)

        for step_num in range(steps_per_epoch):
            print(f"Batch {step_num}")

            batch = batched_ds.skip(step_num).take(1).unbatch()
            files = ds_to_list(batch)

            for bid, fp in enumerate(files):
                file_id = step_num * batch_size + bid
                self.assertEqual(fp, self.list_of_files[file_id])

    def test_test_dataset(self):
        """
        Makes sure that test data are only to be found in test_ds
        and not in the train/val datasets.
        """
        for f in ds_to_list(self.test_ds):
            self.assertTrue("test_" in f)

        for f in ds_to_list(self.config.training_ds):
            self.assertFalse("test_" in f)

        for f in ds_to_list(self.config.validation_ds):
            self.assertFalse("test_" in f)
