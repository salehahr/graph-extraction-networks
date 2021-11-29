import unittest

from tools.data import get_next_filepaths_from_ds, get_skeletonised_ds
from tools.files import get_random_video_path

img_length = 256
base_path = f"/graphics/scratch/schuelej/sar/data/{img_length}"
video_path = get_random_video_path(base_path)


class TestDataGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # small dataset
        cls.num_labels = 15
        cls.batch_size = 3
        cls.steps_per_epoch = int(cls.num_labels / cls.batch_size)

        cls.ds = get_skeletonised_ds(base_path, shuffle=True, seed=13).take(
            cls.num_labels
        )
        print(f"Using {cls.num_labels} images.")

        cls.train_ds, cls.val_ds = cls.split_data()

        # so that the iteration does not go beyond num_labels
        cls.ds = list(cls.ds.as_numpy_iterator())
        cls.train_ds = list(cls.train_ds.as_numpy_iterator())
        cls.val_ds = list(cls.val_ds.as_numpy_iterator())

    @classmethod
    def split_data(cls):
        validation_fraction = 0.1
        num_validation = int(validation_fraction * cls.num_labels)
        num_train = cls.num_labels - num_validation

        train_ds = cls.ds.take(num_train)
        val_ds = cls.ds.skip(num_train)

        return train_ds, val_ds

    def test_split_data(self):
        self.assertEqual(len(self.ds), len(self.train_ds) + len(self.val_ds))

    # def test_get_filepaths(self):
    #     num_val = len(self.val_ds)
    #
    #     print(f'Getting filepaths from validation DS with {num_val} files.')
    #     for i in range(num_val * 2):
    #         fp, graph_fp = get_next_filepaths_from_ds(self.val_ds)
    #         print(fp)
    #         self.assertTrue(os.path.isfile(fp))
    #         self.assertTrue(os.path.isfile(graph_fp))

    def test_get_batch_filepaths(self):
        filepaths = [
            get_next_filepaths_from_ds(self.ds) for i in range(self.batch_size)
        ]
        skel_fps, graph_fps = zip(*filepaths)

        print(skel_fps)
        print(graph_fps)
