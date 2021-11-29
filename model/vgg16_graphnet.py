# https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py
# -*- coding: utf-8 -*-
"""VGG16 model for Keras.
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import print_function

import warnings

import numpy as np
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.engine.topology import get_source_inputs
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, GlobalMaxPooling1D
from keras.preprocessing import image

# from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

from model.utils import callback


class GraphNet_vgg16(Model):
    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    """

    def __init__(
        self, input_size=[256, 256, 2], pretrained_weights=None, max_nr_nodes=128
    ):

        self.input_size = (input_size,)
        self.pretrained_weights = (pretrained_weights,)
        self.max_nr_nodes = max_nr_nodes

        # dimensions of adjacency matrix
        adj_dim = int((max_nr_nodes * max_nr_nodes - max_nr_nodes) / 2)

        # Block 1
        input = Input(shape=input_size, name="input_image")
        x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(
            input
        )
        x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(
            x
        )
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(
            x
        )
        x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(
            x
        )
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(
            x
        )
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(
            x
        )
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(
            x
        )
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(
            x
        )
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(
            x
        )
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(
            x
        )
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(
            x
        )
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(
            x
        )
        x = Conv2D(
            adj_dim, (3, 3), activation="sigmoid", padding="same", name="block5_conv3"
        )(x)
        # x = MaxPooling2D(,(2, 2), strides=(2, 2), name='block5_pool')(x)

        output = GlobalMaxPooling2D(data_format="channels_last")(x)
        # model = Model(inputs, x, name='graph_vgg16')

        # initialize Keras Model with defined above input and output layers
        super(GraphNet_vgg16, self).__init__(inputs=input, outputs=output)
        # load preatrained weights
        if pretrained_weights:
            self.load_weights(pretrained_weights)

    def build(self):
        self.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        self.summary()

    def save_model(self, name):
        self.save_weights(name)

    @staticmethod
    def checkpoint(checkpoint_callback_name):
        return callback(checkpoint_callback_name)


# TODO: FIX SAVING MODEL: AT THIS POINT, ONLY SAVING MODEL WEIGHTS IS AVAILBILE
# SINCE SUBSCLASSING FROM KERAS.MODEL RESTRICTS SAVING MODEL AS AN HDF5 FILE
