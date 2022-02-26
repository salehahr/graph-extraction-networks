import os
from typing import Tuple, Union

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model.utils import double_conv, input_tensor, pooling, single_conv


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

        x = input_tensor(input_size)
        out = self._conv2_blocks(x)
        out = self._conv3_blocks(out)

        out = GlobalMaxPooling2D(data_format="channels_last", keepdims=True)(out)
        out = tf.keras.layers.Conv1D(1, 1, activation="sigmoid")(out)
        out = tf.squeeze(out, name="adj_output", axis=[1, 2])  # shape: (None, 1)

        # initialize Keras Model with defined above input and output layers
        super(VGG16, self).__init__(inputs=x, outputs=out, name="EdgeNN")

        # load pretrained weights
        if pretrained_weights is not None and os.path.isfile(pretrained_weights):
            self.load_weights(pretrained_weights)

    def _conv2_blocks(self, x: tf.Tensor):
        # Default: Blocks 1, 2
        for i in range(1, self.n_conv2_blocks + 1):
            x = double_conv(
                x,
                self.n_filters * 2 ** (i - 1),
                f"relu_C2_block{i}",
                normalise=True,
            )
            x = pooling(x, dropout_rate=0)

        return x

    def _conv3_blocks(self, x: tf.Tensor):
        # Default: Blocks 3, 4, 5
        final_block_num = self.n_conv2_blocks + self.n_conv3_blocks

        for i in range(self.n_conv2_blocks + 1, final_block_num + 1):
            x = double_conv(x, self.n_filters * 2 ** (i - 1), name=None, normalise=True)
            x = single_conv(
                x,
                self.n_filters * 2 ** (i - 1),
                kernel_size=3,
                activation="relu",
                name=f"relu_C3_block{i}",
                padding=True,
            )
            x = pooling(x, dropout_rate=0)

        return x

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
