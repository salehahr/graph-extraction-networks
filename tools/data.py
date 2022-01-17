import os

from tensorflow.data import Dataset


def get_skeletonised_ds(data_path: str, seed: int, is_test: bool = False) -> Dataset:
    """
    Returns skeletonised image paths.
    :param data_path: folder where the data is stored
    :param seed: seed for shuffling function, set None for random
    :param is_test: True if test set filepaths are to be used
    :return: the skeletonised image paths as a Dataset object
    """
    if is_test:
        skeletonised_files_glob = [
            os.path.join(data_path, "test_*/skeleton/*.png"),
            os.path.join(data_path, "*/test_*/skeleton/*.png"),
        ]
    else:
        skeletonised_files_glob = [
            os.path.join(data_path, "[!t]*/skeleton/*.png"),
            os.path.join(data_path, "*/[!t]*/skeleton/*.png"),
        ]

    ds = Dataset.list_files(skeletonised_files_glob, shuffle=False)

    return ds.shuffle(len(ds), seed=seed, reshuffle_each_iteration=False)


def ds_to_list(dataset: Dataset) -> list:
    return [f.decode("utf-8") for f in dataset.as_numpy_iterator()]
