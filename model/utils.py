from typing import List, Optional, Tuple, Union

import tensorflow as tf
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    Layer,
    MaxPooling2D,
    concatenate,
)

""" utils.py
File that defines layers for u-net model.
1. input_tensor - Input layer
2. single_conv - one 2D Convolutional layer
3. double_conv - two Sequential 2D Convolutional layers
4. deconv - one 2D Transposed Convolutional layer
5, pooling - one Max Pooling layer followed by Dropout function
6. merge - concatenates two layers
7. callback - returns a ModelCheckpoint, used in main.py for model fitting
"""


# function that defines input layers for given shape
def input_tensor(
    input_size: Union[Tuple[int, int, int], Tuple[Tuple[int, int], int]],
    name: str = "input",
) -> Input:
    x = Input(input_size, name=name)
    return x


# function that defines one convolutional layer with certain number of filters
def single_conv(
    input_tensor,
    n_filters: int,
    kernel_size: int,
    name: str,
    activation: Optional[str],
    padding: Optional[bool],
    normalise: bool = True,
    use_logits: bool = False,
):
    x = Conv2D(
        name=name if use_logits is True else None,
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        padding="same" if padding is True else "valid",  # default keras
    )(input_tensor)

    if normalise:
        x = BatchNormalization()(x)

    if use_logits:
        return x
    else:
        return Activation(activation, name=name)(x)


# function that defines two sequential 2D convolutional layers with certain number of filters
def double_conv(
    input_tensor,
    n_filters: int,
    name: str,
    kernel_size: int = 3,
    normalise: bool = True,
    activation: str = "relu",
    concat_value=None,
):
    x = input_tensor
    for i in range(2):
        x = Conv2D(
            filters=n_filters,
            kernel_size=(kernel_size, kernel_size),
            padding="same",
            kernel_initializer="he_normal",
            name=f"conv2d_{i}_{name}",
        )(x)

        if normalise:
            x = BatchNormalization(name=f"bn{i}_{name}")(x)

        if concat_value is not None:
            x = concatenate([x, concat_value], axis=-1)

        x = Activation(activation, name=f"relu{i}_{name}")(x)

    return x


def pre_output_conv(input_tensor, n_filters, name, kernel_size=2):
    x = Conv2D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        padding="same",
        kernel_initializer="he_normal",
    )(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu", name=name)(x)
    return x


# function that defines 2D transposed convolutional (Deconvolutional) layer
def deconv(input_tensor, n_filters, kernel_size=3, stride=2):
    x = Conv2DTranspose(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        strides=(stride, stride),
        padding="same",
    )(input_tensor)
    return x


# function that defines Max Pooling layer with pool size 2 and applies Dropout
def pooling(input_tensor, dropout_rate=0.1):
    x = MaxPooling2D(pool_size=(2, 2))(input_tensor)

    if dropout_rate > 0:
        x = Dropout(rate=dropout_rate)(x)

    return x


# function that merges two layers (Concatenate)
def merge(input1, input2):
    x = concatenate([input1, input2])
    return x


def sum_across(inputs: List, name: Optional[str] = None):
    return SummationLayer(name=name)(inputs)


def extract_combo(input_, begin, size, name: Optional[str] = None):
    return ComboExtraction(begin, size, name=name)(input_)


class SummationLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(SummationLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        concat_inputs = tf.concat(inputs, axis=-1)
        return tf.math.reduce_sum(concat_inputs, axis=-1, keepdims=True)


class ComboExtraction(Layer):
    def __init__(self, begin, size, **kwargs):
        super(ComboExtraction, self).__init__(**kwargs)
        self.begin = begin
        self.size = size

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "begin": self.begin,
                "size": self.size,
            }
        )
        return config

    def call(self, input_, **kwargs):
        return tf.slice(input_, self.begin, self.size)
