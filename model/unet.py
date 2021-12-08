from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model.utils import (
    callback,
    deconv,
    double_conv,
    input_tensor,
    merge,
    pooling,
    single_conv,
)


class UNet(Model):
    def __init__(self, input_size: int, n_filters: int, pretrained_weights=None):
        # define input layer
        x = input_tensor(input_size)  # 256x256x3

        # contractive path
        self.skips = self._contractive_path(x, n_filters)

        # expansive path
        self.final_layer = self._expansive_path(self.skips, n_filters)

        # define output layers
        self.node_pos = single_conv(
            self.final_layer, 1, 1, name="node_pos", activation="sigmoid"
        )
        self.degrees = single_conv(
            self.final_layer, 1, 1, name="degrees", activation="softmax"
        )
        self.node_types = single_conv(
            self.final_layer, 1, 1, name="node_types", activation="softmax"
        )

        # initialize Keras Model with defined above input and output layers
        super(UNet, self).__init__(
            inputs=x, outputs=[self.node_pos, self.degrees, self.node_types]
        )

        # load preatrained weights
        if pretrained_weights:
            self.load_weights(pretrained_weights)

    def build(self):
        self.compile(
            optimizer=Adam(),
            loss={
                "node_pos": "binary_crossentropy",
                "degrees": "categorical_crossentropy",
                "node_types": "categorical_crossentropy",
            },
            metrics=["accuracy"],
        )
        self.summary()

    @staticmethod
    def _contractive_path(x, n_filters: int) -> list:
        # contraction path
        conv1 = double_conv(x, n_filters * 1, "relu_block_1")  # 64
        pool1 = pooling(conv1)

        conv2 = double_conv(pool1, n_filters * 2, "relu_block_2")  # 128
        pool2 = pooling(conv2)

        conv3 = double_conv(pool2, n_filters * 4, "relu_block_3")  # 256
        pool3 = pooling(conv3)

        conv4 = double_conv(pool3, n_filters * 8, "relu_block_4")  # 512
        pool4 = pooling(conv4)

        conv5 = double_conv(pool4, n_filters * 16, "relu_block_5")  # 1024

        return [conv1, conv2, conv3, conv4, conv5]

    @staticmethod
    def _expansive_path(skips, n_filters: int):
        conv1, conv2, conv3, conv4, conv5 = skips

        # expansive path
        up6 = deconv(conv5, n_filters * 8)  # 512
        up6 = merge(conv4, up6)
        conv6 = double_conv(up6, n_filters * 8, "relu_block_4r")  #

        up7 = deconv(conv6, n_filters * 4)
        up7 = merge(conv3, up7)
        conv7 = double_conv(up7, n_filters * 4, "relu_block_3r")

        up8 = deconv(conv7, n_filters * 2)
        up8 = merge(conv2, up8)
        conv8 = double_conv(up8, n_filters * 2, "relu_block_2r")

        up9 = deconv(conv8, n_filters * 1)
        up9 = merge(conv1, up9)

        return double_conv(up9, n_filters * 1, "relu_block_1r")  # 256x256x64

    @staticmethod
    def checkpoint(name):
        return callback(name)

    @staticmethod
    def tensorboard_callback(name):
        return TensorBoard(log_dir=name, histogram_freq=1)
