import unittest

import tensorflow as tf

from model.unet import NodesNN, UNet
from tools import Config, DataGenerator, TestType


class TestUNetLayers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        cls.network = cls.config.network.node_extraction

        cls.training_ds = DataGenerator(cls.config, cls.network, TestType.TRAINING)
        cls.input_size = (*cls.config.img_dims, cls.network.input_channels)
        cls.num_filters = 4
        cls.model = cls._init_model()

        cls.input_layer = cls.model.get_layer("input").output

    @classmethod
    def _init_model(cls) -> UNet:
        unet = UNet(
            input_size=(*cls.config.img_dims, cls.network.input_channels),
            n_filters=cls.num_filters,
        )
        unet.build()

        return unet

    def test_input_layer(self):
        self.assertEqual(self.input_layer.dtype, tf.float32)
        self.assertEqual(self.input_layer.shape[1:], (256, 256, 1))

    def test_contractive_path(self):
        conv5 = self.model.skips[-1]

        self.assertEqual(conv5.dtype, tf.float32)
        self.assertEqual(conv5.shape[1:], (16, 16, 64))

    def test_final_layer(self):
        self.assertEqual(self.model.final_layer.dtype, tf.float32)
        self.assertEqual(self.model.final_layer.shape[1:], (256, 256, self.num_filters))


class TestNodeAttributeLayers(TestUNetLayers):
    @classmethod
    def _init_model(cls) -> NodesNN:
        nodes_nn = NodesNN(
            input_size=(*cls.config.img_dims, cls.network.input_channels),
            n_filters=cls.num_filters,
        )
        nodes_nn.build()

        return nodes_nn

    def test_node_pos_output(self):
        self.assertEqual(self.model.node_pos.dtype, tf.float32)
        self.assertEqual(self.model.node_pos.shape[1:], (256, 256, 1))
