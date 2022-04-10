from typing import Dict, List, Tuple, Union

import tensorflow as tf

from model.utils import (
    deconv,
    double_conv,
    input_tensor,
    merge,
    pooling,
    pre_output_conv,
    single_conv,
)


class UNet(tf.keras.models.Model):
    def __init__(
        self,
        input_size: Union[Tuple[int, int, int], Tuple[Tuple[int, int], int]],
        n_filters: int,
        depth: int = 5,
        pretrained_weights=None,
        eager: bool = True,
    ):
        tf.config.run_functions_eagerly(eager)

        # define input layer
        x = input_tensor(input_size)  # 256x256x3

        # contractive path
        self.skips = self._contractive_path(x, n_filters, depth)

        # expansive path
        self.final_layer = self._expansive_path(self.skips, n_filters, depth)

        # initialize Keras Model with defined above input and output layers
        super(UNet, self).__init__(inputs=x, outputs=self._get_outputs())

        # load pretrained weights
        if pretrained_weights:
            self.load_weights(pretrained_weights)

    def _get_outputs(self):
        return self.final_layer

    def _get_loss(self):
        return {"final": "sparse_categorical_crossentropy"}

    def _define_metrics(self) -> Dict[str, List[tf.keras.metrics.Metric]]:
        return {"final": [tf.keras.metrics.Accuracy(name="accuracy")]}

    def build(self, **kwargs):
        self.recompile(**kwargs)

    def recompile(self, **kwargs):
        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=self._get_loss(),
            metrics=self._define_metrics(),
            **kwargs,
        )

    @staticmethod
    def _contractive_path(x, n_filters: int, depth: int) -> list:
        skips = []
        pool = None

        for i in range(1, depth + 1):
            conv_input = pool if i != 1 else x

            conv = double_conv(conv_input, n_filters * 2 ** (i - 1), f"relu_block_{i}")
            pool = pooling(conv)

            skips.append(conv)

        return skips

    @staticmethod
    def _expansive_path(skips, n_filters: int, depth: int):
        up = None
        conv = None
        skips_copy = skips.copy()

        for i in range(depth, 1, -1):
            up_input = skips_copy.pop() if i == depth else conv

            up = deconv(up_input, n_filters * 2 ** (i - 2))  # 512
            up = merge(skips_copy.pop(), up)
            conv = double_conv(up, n_filters * 2 ** (i - 2), f"relu_block_{i - 1}r")  #

        return double_conv(up, n_filters * 1, "relu_block_1r")  # 256x256x64

    @staticmethod
    def checkpoint(filepath, save_frequency="epoch"):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor="epoch_loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            save_freq=save_frequency,
        )

    @staticmethod
    def tensorboard_callback(filepath):
        return tf.keras.callbacks.TensorBoard(
            log_dir=filepath, histogram_freq=1, update_freq="epoch"
        )


class NodesNN(UNet):
    def __init__(
        self,
        input_size: Union[Tuple[int, int, int], Tuple[Tuple[int, int], int]],
        n_filters: int,
        depth: int = 5,
        pretrained_weights=None,
        eager: bool = True,
    ):
        self.node_pos = None
        self.degrees = None
        self.node_types = None

        super(NodesNN, self).__init__(
            input_size, n_filters, depth, pretrained_weights, eager
        )

    def _get_outputs(self):
        self.node_pos = single_conv(
            self.final_layer,
            1,
            1,
            name="node_pos",
            activation="sigmoid",
            normalise=False,
            padding=None,
        )
        self.degrees = single_conv(
            self.final_layer,
            5,
            1,
            name="degrees",
            activation="softmax",
            normalise=False,
            padding=None,
        )
        self.node_types = single_conv(
            self.final_layer,
            4,
            1,
            name="node_types",
            activation="softmax",
            normalise=False,
            padding=None,
        )

        return [self.node_pos, self.degrees, self.node_types]

    def _get_loss(self):
        return {
            "node_pos": "binary_crossentropy",
            "degrees": "sparse_categorical_crossentropy",
            "node_types": "sparse_categorical_crossentropy",
        }

    def _define_metrics(self) -> Dict[str, List[tf.keras.metrics.Metric]]:
        return {
            "node_pos": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.BinaryCrossentropy(name="ce"),
                tf.keras.metrics.KLDivergence(name="kl"),
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
                tf.keras.metrics.MeanSquaredLogarithmicError(name="msle"),
                tf.keras.metrics.RootMeanSquaredError(name="rms"),
            ],
            "degrees": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseCategoricalCrossentropy(name="ce"),
                tf.keras.metrics.KLDivergence(name="kl"),
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
                tf.keras.metrics.MeanSquaredLogarithmicError(name="msle"),
                tf.keras.metrics.RootMeanSquaredError(name="rms"),
            ],
            "node_types": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseCategoricalCrossentropy(name="ce"),
                tf.keras.metrics.KLDivergence(name="kl"),
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
                tf.keras.metrics.MeanSquaredLogarithmicError(name="msle"),
                tf.keras.metrics.RootMeanSquaredError(name="rms"),
            ],
        }


class NodesNNExtended(NodesNN):
    """With additional filters between the base U-Net and the node attribute outputs."""

    def __init__(self, *args, **kwargs):
        super(NodesNN, self).__init__(*args, **kwargs)

    def _get_outputs(self):
        pre_node_pos = pre_output_conv(self.final_layer, 6, name="pre_node_pos")
        pre_degrees = pre_output_conv(self.final_layer, 6, name="pre_degrees")
        pre_node_types = pre_output_conv(self.final_layer, 6, name="pre_node_types")

        self.node_pos = single_conv(
            pre_node_pos,
            1,
            1,
            name="node_pos",
            activation="sigmoid",
            normalise=False,
            padding=None,
        )
        self.degrees = single_conv(
            pre_degrees,
            5,
            1,
            name="degrees",
            activation="softmax",
            normalise=False,
            padding=None,
        )
        self.node_types = single_conv(
            pre_node_types,
            4,
            1,
            name="node_types",
            activation="softmax",
            normalise=False,
            padding=None,
        )

        return [self.node_pos, self.degrees, self.node_types]
