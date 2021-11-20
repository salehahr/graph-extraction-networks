from keras.preprocessing.image import ImageDataGenerator
from tools.image import normalize_mask
import skimage.io as io
import os

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

        img_filt = (filt * 255).astype('uint8')
        img_skel = (binary_img * 255).astype('uint8')

        io.imsave(os.path.join(save_path, f"{i:d}_predict_filt.png"), img_filt)
        io.imsave(os.path.join(save_path, f"{i:d}_predict_skel.png"), img_skel)
