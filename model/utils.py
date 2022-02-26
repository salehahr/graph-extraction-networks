from typing import Optional, Tuple

from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
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
def input_tensor(input_size: Tuple[int, int, int]) -> Input:
    x = Input(input_size, name="input")
    return x


# function that defines one convolutional layer with certain number of filters
def single_conv(
    input_tensor,
    n_filters: int,
    kernel_size: int,
    name: str,
    activation: str,
    padding: bool,
):
    return Conv2D(
        name=name,
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        activation=activation,
        padding="same" if padding is True else None,
    )(input_tensor)


# function that defines two sequential 2D convolutional layers with certain number of filters
def double_conv(
    input_tensor,
    n_filters: int,
    name: Optional[str],
    kernel_size: int = 3,
    normalise: bool = True,
):
    x = Conv2D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        padding="same",
        kernel_initializer="he_normal",
    )(input_tensor)

    if normalise:
        x = BatchNormalization()(x)

    x = Activation("relu")(x)
    x = Conv2D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        padding="same",
        kernel_initializer="he_normal",
    )(x)

    if normalise:
        x = BatchNormalization()(x)

    x = Activation("relu", name=name)(x)
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
