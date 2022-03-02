import os
from typing import Optional, Tuple, Union

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model.utils import (
    double_conv,
    extract_combo,
    input_tensor,
    pooling,
    single_conv,
    sum_across,
)
from tools.timer import timer


class VGG16(Model):
    def __init__(
        self,
        input_size: Union[Tuple[int, int, int], Tuple[Tuple[int, int], int]],
        n_filters: int,
        n_conv2_blocks: int = 2,
        n_conv3_blocks: int = 3,
        pretrained_weights: Optional[str] = None,
        learning_rate: float = 0.01,
        optimiser: str = "Adam",
    ):
        self.n_filters = n_filters
        self.n_conv2_blocks = n_conv2_blocks
        self.n_conv3_blocks = n_conv3_blocks

        self.learning_rate = learning_rate
        self.optimiser = self._set_optimiser(optimiser)

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
            self.pretrained = True
            self.weights_path = pretrained_weights
            self.load_weights(pretrained_weights)
        else:
            self.pretrained = False
            self.weights_path = None

    def _set_optimiser(self, optimiser: str):
        optimiser = optimiser.lower()

        if optimiser == "adam":
            return Adam(learning_rate=self.learning_rate)
        elif optimiser == "radam":
            return tfa.optimizers.RectifiedAdam(learning_rate=self.learning_rate)

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
        # self.summary()

    def recompile(self, **kwargs):
        metrics = [
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
        ]

        self.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=metrics,
            run_eagerly=False,
            **kwargs,
        )


class EdgeNN(VGG16):
    def __init__(
        self,
        input_size: Union[Tuple[int, int, int], Tuple[Tuple[int, int], int]],
        n_filters: int,
        n_conv2_blocks: int = 2,
        n_conv3_blocks: int = 3,
        pretrained_weights: Optional[str] = None,
        learning_rate: float = 0.01,
        optimiser: str = "Adam",
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
            optimiser,
        )

    @tf.function
    def __call__(self, *args, **kwargs):
        print("tracing EdgeNN.__call__ ...")
        return super(EdgeNN, self).__call__(*args, **kwargs)

    @timer
    def predict(self, *args, **kwargs):
        return super(EdgeNN, self).predict(*args, **kwargs)

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
        # combo = None

        # Default: Blocks 1, 2
        for i in range(1, self.n_conv2_blocks + 1):
            node_pair = self._node_pair if i == 1 else None  # combo

            x = double_conv(
                x,
                self.n_filters * 2 ** (i - 1),
                f"block{i}",
                normalise=True,
                concat_value=node_pair,
            )
            x = pooling(x, dropout_rate=0)
            # combo = extract_combo(x, [0, 0, 0, x.get_shape()[-1] - 1], [-1, -1, -1, 1])

        self._conv2_output = x
