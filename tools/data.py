from keras.preprocessing.image import ImageDataGenerator
from tools.image import normalize_mask
import skimage.io as io
import os
import skimage.io as io

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.data import Dataset

from tools.image import normalize_mask, classifier_preview

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def train_generator(
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    target_size,
    image_color_mode = 'grayscale',
    mask_color_mode = 'grayscale'
):
    """ Image Data Generator
    Function that generates batches of data (img, mask) for training
    from specified folder. Returns images with specified pixel size
    Does preprocessing (normalization to 0-1)
    """
    # no augmentation, only rescaling
    image_datagen = ImageDataGenerator(rescale=1. / 255)
    mask_datagen = ImageDataGenerator(rescale=1. / 255)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        seed = 1
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        seed = 1
    )
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        mask = normalize_mask(mask)
        yield (img,mask)


def save_results(
    save_path,
    results
):
    """ Save Results
    Function that takes predictions from U-Net model
    and saves them to specified folder.
    """
    results_filtered = results[0]
    results_skeleton = results[1]

    try:
        print(results_filtered.shape)
    except:
        print(len(results_filtered))

    for i, filt in enumerate(results_filtered):
        binary_img = normalize_mask(results_skeleton[i, :, :, 0])

def get_skeletonised_ds(data_path: str, shuffle: bool, seed: int) -> Dataset:
    skeletonised_files_glob = [
        os.path.join(data_path, '**/skeleton/*.png'),
        os.path.join(data_path, '**/**/skeleton/*.png'),
        ]
    return Dataset.list_files(skeletonised_files_glob, shuffle=shuffle, seed=seed)


def get_next_filepaths_from_ds(dataset: Dataset):
    skel_fp = next(iter(dataset)).numpy().decode("utf-8")
    graph_fp = skel_fp.replace('skeleton', 'graphs').replace('.png', '.json')
    return skel_fp, graph_fp
