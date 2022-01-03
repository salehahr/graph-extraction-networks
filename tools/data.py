import os

import skimage.io as io
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.data import Dataset

from tools.image import classifier_preview, classify

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def train_generator(
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    target_size,
    image_color_mode="grayscale",
    mask_color_mode="grayscale",
):
    """Image Data Generator
    Function that generates batches of data (img, mask) for training
    from specified folder. Returns images with specified pixel size
    Does preprocessing (normalization to 0-1)
    """
    # no augmentation, only rescaling
    image_datagen = ImageDataGenerator(rescale=1.0 / 255)
    mask_datagen = ImageDataGenerator(rescale=1.0 / 255)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=1,
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=1,
    )
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        mask, _ = classify(mask)
        yield (img, mask)


def get_skeletonised_ds(data_path: str, seed: int, is_test: bool = False) -> Dataset:
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


def get_next_filepaths_from_ds(dataset: Dataset):
    skel_fp = next(iter(dataset)).numpy().decode("utf-8")
    graph_fp = skel_fp.replace("skeleton", "graphs").replace(".png", ".json")
    return skel_fp, graph_fp
