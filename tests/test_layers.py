import math
import unittest

import numpy as np
import tensorflow as tf

from model.utils import deconv, double_conv, input_tensor, merge, pooling, single_conv
from tools import Config, DataGenerator, TestType


class TestLayers(unittest.TestCase):
    """Has no coverage (reimplementing functions in UNet)."""

    @classmethod
    def setUpClass(cls) -> None:
        config = Config("test_config.yaml")
        cls.training_ds = DataGenerator(config, TestType.TRAINING)
        cls.input_size = (*config.img_dims, config.input_channels)
        cls.num_filters = 64

    def _get_input_layer(self):
        return input_tensor(self.input_size)

    def _get_cp_1(self, x):
        conv1 = double_conv(x, self.num_filters * 1)
        return pooling(conv1)

    def _get_contraction_path(self):
        x = self._get_input_layer()

        # contraction path
        conv1 = double_conv(x, self.num_filters * 1)  # 64
        pool1 = pooling(conv1)

        conv2 = double_conv(pool1, self.num_filters * 2)  # 128
        pool2 = pooling(conv2)

        conv3 = double_conv(pool2, self.num_filters * 4)  # 256
        pool3 = pooling(conv3)

        conv4 = double_conv(pool3, self.num_filters * 8)  # 512
        pool4 = pooling(conv4)

        return double_conv(pool4, self.num_filters * 16)  # 1024

    def _get_penultimate_layer(self):
        x = self._get_input_layer()

        # contraction path
        conv1 = double_conv(x, self.num_filters * 1)  # 64
        pool1 = pooling(conv1)

        conv2 = double_conv(pool1, self.num_filters * 2)  # 128
        pool2 = pooling(conv2)

        conv3 = double_conv(pool2, self.num_filters * 4)  # 256
        pool3 = pooling(conv3)

        conv4 = double_conv(pool3, self.num_filters * 8)  # 512
        pool4 = pooling(conv4)

        conv5 = double_conv(pool4, self.num_filters * 16)  # 1024

        # expansive path
        up6 = deconv(conv5, self.num_filters * 8)  # 512
        up6 = merge(conv4, up6)
        conv6 = double_conv(up6, self.num_filters * 8)  #

        up7 = deconv(conv6, self.num_filters * 4)
        up7 = merge(conv3, up7)
        conv7 = double_conv(up7, self.num_filters * 4)

        up8 = deconv(conv7, self.num_filters * 2)
        up8 = merge(conv2, up8)
        conv8 = double_conv(up8, self.num_filters * 2)

        up9 = deconv(conv8, self.num_filters * 1)
        up9 = merge(conv1, up9)
        return x, double_conv(up9, self.num_filters * 1)

    def test_input_layer(self):
        x = self._get_input_layer()

        self.assertEqual(x.dtype, tf.float32)
        self.assertEqual(x.shape[1:], (256, 256, 1))

    def test_contract_cp_1(self):
        x = self._get_input_layer()
        cp_1 = self._get_cp_1(x)

        self.assertEqual(cp_1.dtype, tf.float32)
        self.assertEqual(cp_1.shape[1:], (128, 128, 64))

    def test_contractive_path(self):
        conv5 = self._get_contraction_path()

        self.assertEqual(conv5.dtype, tf.float32)
        self.assertEqual(conv5.shape[1:], (16, 16, 1024))

    def test_penultimate_layer(self):
        _, pu_layer = self._get_penultimate_layer()

        self.assertEqual(pu_layer.dtype, tf.float32)
        self.assertEqual(pu_layer.shape[1:], (256, 256, 64))

    def test_node_pos_output(self):
        pu_layer = self._get_penultimate_layer()
        node_pos = single_conv(pu_layer, 1, 1, name="node_pos", activation="relu")

        self.assertEqual(node_pos.dtype, tf.float32)
        self.assertEqual(node_pos.shape[1:], (256, 256, 1))

        return

    def test_losses_in_node_pos_model(self):
        input_layer, pu_layer = self._get_penultimate_layer()
        node_pos = single_conv(pu_layer, 1, 1, name="node_pos", activation="relu")

        model = tf.keras.Model(inputs=input_layer, outputs=[node_pos])
        model.compile(
            optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        skel_input, node_pos_gt, _, _ = self._get_first_images(self.training_ds)

        hist = model.fit(x=skel_input, y=node_pos_gt)
        losses = hist.history["loss"]

        for l in losses:
            self.assertFalse(math.isnan(l))

    def test_losses_in_node_degrees_model(self):
        input_layer, pu_layer = self._get_penultimate_layer()
        node_degrees_layer = single_conv(
            pu_layer, 1, 1, name="node_degrees", activation="softmax"
        )

        model = tf.keras.Model(inputs=input_layer, outputs=[node_degrees_layer])
        model.compile(
            optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        skel_input, _, node_degrees_gt, _ = self._get_first_images(self.training_ds)

        hist = model.fit(x=skel_input, y=node_degrees_gt)
        losses = hist.history["loss"]

        for l in losses:
            self.assertFalse(math.isnan(l))

    @staticmethod
    def _get_first_images(dataset):
        file_id = 0
        x, y = dataset[file_id]

        batch_id = 0
        skel_input = np.expand_dims(x[batch_id], 0)
        node_pos_gt = np.expand_dims(y[0][batch_id], 0)
        node_degrees_gt = np.expand_dims(y[1][batch_id], 0)
        node_types_gt = np.expand_dims(y[2][batch_id], 0)

        return skel_input, node_pos_gt, node_degrees_gt, node_types_gt
