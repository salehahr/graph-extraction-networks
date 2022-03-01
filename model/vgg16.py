import os
from typing import Tuple, Union

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model.utils import double_conv, input_tensor, pooling, single_conv, sum_across


class VGG16(Model):
    def __init__(
        self,
        input_size: Union[Tuple[int, int, int], Tuple[Tuple[int, int], int]],
        n_filters: int,
        n_conv2_blocks: int = 2,
        n_conv3_blocks: int = 3,
        pretrained_weights=None,
        learning_rate: float = 0.01,
    ):
        self.n_filters = n_filters
        self.n_conv2_blocks = n_conv2_blocks
        self.n_conv3_blocks = n_conv3_blocks

        self.learning_rate = learning_rate

        # intermediate outputs
        self._input: tf.Tensor
        self._conv2_output: tf.Tensor
        self._conv3_output: tf.Tensor
        self._output: tf.Tensor

        self._build_layers(input_size)

        # initialize Keras Model with defined above input and output layers
        super(VGG16, self).__init__(
            inputs=self._input, outputs=self._output, name="EdgeNN"
        )

        # load pretrained weights
        if pretrained_weights is not None and os.path.isfile(pretrained_weights):
            self.load_weights(pretrained_weights)

    def _build_layers(self, input_size):
        self._set_input(input_size)
        self._set_conv2_blocks()
        self._set_conv3_blocks()
        self._set_output_block()

    def _set_input(
        self, input_size: Union[Tuple[int, int, int], Tuple[Tuple[int, int], int]]
    ) -> None:
        self._input = input_tensor(input_size)

    def _set_conv2_blocks(self) -> None:
        x = self._input

        # Default: Blocks 1, 2
        for i in range(1, self.n_conv2_blocks + 1):
            x = double_conv(
                x,
                self.n_filters * 2 ** (i - 1),
                f"block{i}",
                normalise=True,
            )
            x = pooling(x, dropout_rate=0)

        self._conv2_output = x

    def _set_conv3_blocks(self) -> None:
        x = self._conv2_output

        # Default: Blocks 3, 4, 5
        final_block_num = self.n_conv2_blocks + self.n_conv3_blocks

        for i in range(self.n_conv2_blocks + 1, final_block_num + 1):
            x = double_conv(
                x, self.n_filters * 2 ** (i - 1), f"block{i}", normalise=True
            )
            x = single_conv(
                x,
                self.n_filters * 2 ** (i - 1),
                kernel_size=3,
                activation="relu",
                name=f"relu_C3_block{i}",
                padding=True,
                normalise=True,
            )
            x = pooling(x, dropout_rate=0)

        self._conv3_output = x

    def _set_output_block(self) -> None:
        x = self._conv3_output

        out = GlobalMaxPooling2D(data_format="channels_last", keepdims=True)(x)
        out = tf.keras.layers.Conv1D(1, 1, activation="sigmoid")(out)
        out = tf.squeeze(out, name="adj_output", axis=[1, 2])  # shape: (None, 1)

        self._output = out

    def build(self, **kwargs):
        self.recompile(**kwargs)
        self.summary()

    def recompile(self, **kwargs):
        self.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
            **kwargs,
        )

    @staticmethod
    def checkpoint(filepath, save_frequency="epoch"):
        return ModelCheckpoint(
            filepath,
            monitor="epoch_loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            save_freq=save_frequency,
        )


class EdgeNN(VGG16):
    def __init__(
        self,
        input_size: Union[Tuple[int, int, int], Tuple[Tuple[int, int], int]],
        n_filters: int,
        n_conv2_blocks: int = 2,
        n_conv3_blocks: int = 3,
        pretrained_weights=None,
        learning_rate: float = 0.01,
    ):
        self._input_sum: tf.Tensor
        self._node_pair: tf.Tensor

        super(EdgeNN, self).__init__(
            input_size,
            n_filters,
            n_conv2_blocks,
            n_conv3_blocks,
            pretrained_weights,
            learning_rate,
        )

    def _build_layers(self, input_size):
        self._set_input(input_size)
        self._set_conv2_blocks()
        self._set_conv3_blocks()
        self._set_output_block()

    def _set_input(
        self, input_size: Union[Tuple[int, int, int], Tuple[Tuple[int, int], int]]
    ) -> None:
        skel_img = input_tensor(input_size, name="skel_img")
        node_pos = input_tensor(input_size, name="node_pos")
        node_pair = input_tensor(input_size, name="node_pair")

        self._input = [skel_img, node_pos, node_pair]

        input_sum = sum_across([skel_img, node_pos, node_pair], name="summation")

        self._input_sum = input_sum
        self._node_pair = node_pair

    def _set_conv2_blocks(self) -> None:
        x = self._input_sum

        # Default: Blocks 1, 2
        for i in range(1, self.n_conv2_blocks + 1):
            node_pair = self._node_pair if i == 1 else None

            x = double_conv(
                x,
                self.n_filters * 2 ** (i - 1),
                f"block{i}",
                normalise=True,
                concat_value=node_pair,
            )
            x = pooling(x, dropout_rate=0)

        self._conv2_output = x
