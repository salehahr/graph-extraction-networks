from typing import Tuple

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from wandb.keras import WandbCallback

from model.utils import double_conv, input_tensor, pooling, single_conv


class VGG16(Model):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        n_filters: int,
        pretrained_weights=None,
    ):
        x = input_tensor(input_size)
        out = None

        # Blocks 1, 2
        for i in range(1, 2 + 1):
            conv_input = x if i == 1 else out

            out = double_conv(conv_input, n_filters * 2 ** (i - 1), f"relu_block{i}")
            out = pooling(out)

        # Blocks 3, 4, 5
        for i in range(3, 5 + 1):
            out = double_conv(out, n_filters * 2 ** (i - 1), name=None)
            out = single_conv(
                out,
                n_filters * 2 ** (i - 1),
                kernel_size=3,
                activation="relu",
                name=f"relu_block{i}",
            )
            out = pooling(out)

        out = GlobalMaxPooling2D(data_format="channels_last", keepdims=True)(out)
        out = tf.keras.layers.Conv1D(1, 1, activation="sigmoid")(out)
        out = tf.squeeze(out, name="adj_output", axis=[1, 2])  # shape: (None, 1)

        # initialize Keras Model with defined above input and output layers
        super(VGG16, self).__init__(inputs=x, outputs=out, name="EdgeNN")

        # load pretrained weights
        if pretrained_weights:
            self.load_weights(pretrained_weights)

    def build(self):
        self.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        self.summary()

    def save_model(self, name):
        self.save_weights(name)

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

    @staticmethod
    def wandb_callback():
        return WandbCallback()
