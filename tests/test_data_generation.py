import os.path
import unittest

import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

from tools.PolyGraph import PolyGraph
from tools.files import get_random_video_path, get_random_image
from tools.image import generate_outputs, classifier_preview
from tools.plots import plot_sample, plot_generated_images, plot_classifier_images

img_length = 256
base_path = f'/graphics/scratch/schuelej/sar/data/{img_length}'
video_path = get_random_video_path(base_path)
data_path = '/graphics/scratch/schuelej/sar/data/256/GRK008/cropped'


class TestDataGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        img_name = get_random_image(video_path)
        print(f'Video path: {video_path}')
        print(f'Image: {img_name}')

        # initialise filepaths
        img_cropped_fp = os.path.join(video_path, f'cropped/{img_name}')
        img_masked_fp = img_cropped_fp.replace('cropped', 'masked')
        img_skeletonised_fp = img_cropped_fp.replace('cropped', 'skeleton')
        graph_fp = img_cropped_fp.replace('cropped', 'graphs').replace('.png', '.json')

        # load images and graph
        img_cropped = img_to_array(load_img(img_cropped_fp), dtype=np.uint8)
        img_masked = img_to_array(load_img(img_masked_fp, grayscale=True), dtype=np.uint8)
        img_skeletonised = img_to_array(load_img(img_skeletonised_fp, grayscale=True), dtype=np.uint8)
        graph = PolyGraph.load(graph_fp)

        # generate masks
        output_matrices = generate_outputs(graph, img_length)
        output_img = classifier_preview(output_matrices, img_skeletonised)

        # only one sample in batch
        cls.samples_skeletonised = np.expand_dims(img_skeletonised, axis=0)
        cls.samples_node_pos = np.expand_dims(output_matrices['node_pos'], axis=0)
        cls.samples_degrees = np.expand_dims(output_matrices['degrees'], axis=0)
        cls.samples_node_types = np.expand_dims(output_matrices['node_types'], axis=0)

        cls.plot_title = os.path.relpath(img_cropped_fp.replace('cropped/', ''),
                                         start=base_path)
        sample_images = {'cropped': img_cropped,
                         'masked': img_masked,
                         'skeleton': img_skeletonised,
                         'node_pos': output_img['node_pos'],
                         'degrees': output_img['degrees'],
                         'node_types': output_img['node_types'],
                         }
        plot_sample(sample_images, cls.plot_title)

    def test_transformed_imgs_and_mask(self):
        data_gen_args = dict(rotation_range=90,
                             horizontal_flip=True,
                             vertical_flip=True)

        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        # image_datagen.fit(images, augment=True, seed=seed)
        # mask_datagen.fit(masks, augment=True, seed=seed)

        in_iter = image_datagen.flow(self.samples_skeletonised,
                                     batch_size=1,
                                     seed=seed)
        out_node_pos_iter = mask_datagen.flow(self.samples_node_pos,
                                              batch_size=1,
                                              seed=seed)
        out_degrees_iter = mask_datagen.flow(self.samples_degrees,
                                             batch_size=1,
                                             seed=seed)
        out_node_types_iter = mask_datagen.flow(self.samples_node_types,
                                             batch_size=1,
                                             seed=seed)

        classifier_iterators = {'node_pos': out_node_pos_iter,
                                'degrees': out_degrees_iter,
                                'node_types': out_node_types_iter}

        base_imgs = plot_generated_images(in_iter, 'input: skeleton', cmap='gray')

        plot_classifier_images(classifier_iterators, base_imgs)
