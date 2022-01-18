from typing import Tuple

from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from wandb.keras import WandbCallback

from model.utils import (
    deconv,
    double_conv,
    input_tensor,
    merge,
    pooling,
    pre_output_conv,
    single_conv,
)


class UNet(Model):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        n_filters: int,
        depth: int = 5,
        pretrained_weights=None,
    ):
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

    def build(self):
        self.compile(
            optimizer=Adam(),
            loss=self._get_loss(),
            metrics=["accuracy"],
        )
        self.summary()

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
        return ModelCheckpoint(
            filepath,
            monitor="epoch_loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            save_freq=save_frequency,
        )

    @staticmethod
    def tensorboard_callback(filepath):
        return TensorBoard(log_dir=filepath, histogram_freq=1, update_freq="epoch")

    @staticmethod
    def wandb_callback():
        return WandbCallback()


class NodesNN(UNet):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        n_filters: int,
        depth: int = 5,
        pretrained_weights=None,
    ):
        self.node_pos = None
        self.degrees = None
        self.node_types = None

        super(NodesNN, self).__init__(input_size, n_filters, depth, pretrained_weights)

    def _get_outputs(self):
        self.node_pos = single_conv(
            self.final_layer, 1, 1, name="node_pos", activation="sigmoid"
        )
        self.degrees = single_conv(
            self.final_layer, 5, 1, name="degrees", activation="softmax"
        )
        self.node_types = single_conv(
            self.final_layer, 4, 1, name="node_types", activation="softmax"
        )

        return [self.node_pos, self.degrees, self.node_types]

    def _get_loss(self):
        return {
            "node_pos": "binary_crossentropy",
            "degrees": "sparse_categorical_crossentropy",
            "node_types": "sparse_categorical_crossentropy",
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
            pre_node_pos, 1, 1, name="node_pos", activation="sigmoid"
        )
        self.degrees = single_conv(
            pre_degrees, 5, 1, name="degrees", activation="softmax"
        )
        self.node_types = single_conv(
            pre_node_types, 4, 1, name="node_types", activation="softmax"
        )

        return [self.node_pos, self.degrees, self.node_types]
