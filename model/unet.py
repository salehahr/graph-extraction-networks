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
    def __init__(self, input_size, n_filters, pretrained_weights=None):
        # define input layer
        input = input_tensor(input_size)  # 512x512x3

        # contraction path
        conv1 = double_conv(input, n_filters * 1)  # 64
        pool1 = pooling(conv1)

        conv2 = double_conv(pool1, n_filters * 2)  # 128
        pool2 = pooling(conv2)

        conv3 = double_conv(pool2, n_filters * 4)  # 256
        pool3 = pooling(conv3)

        conv4 = double_conv(pool3, n_filters * 8)  # 512
        pool4 = pooling(conv4)

        conv5 = double_conv(pool4, n_filters * 16)  # 1024

        # expansive path
        up6 = deconv(conv5, n_filters * 8)  # 512
        up6 = merge(conv4, up6)
        conv6 = double_conv(up6, n_filters * 8)  #

        up7 = deconv(conv6, n_filters * 4)
        up7 = merge(conv3, up7)
        conv7 = double_conv(up7, n_filters * 4)

        up8 = deconv(conv7, n_filters * 2)
        up8 = merge(conv2, up8)
        conv8 = double_conv(up8, n_filters * 2)

        up9 = deconv(conv8, n_filters * 1)
        up9 = merge(conv1, up9)
        conv9 = double_conv(up9, n_filters * 1)  # 512x512x64

        # define output layer
        node_pos = single_conv(conv9, 1, 1, name="node_pos", activation="sigmoid")
        degrees = single_conv(conv9, 1, 1, name="degrees", activation="softmax")
        node_types = single_conv(conv9, 1, 1, name="node_types", activation="softmax")
        output = [node_pos, degrees, node_types]

        # initialize Keras Model with defined above input and output layers
        super(UNet, self).__init__(inputs=input, outputs=output)

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
    def checkpoint(name):
        return callback(name)

    @staticmethod
    def tensorboard_callback(name):
        return TensorBoard(log_dir=name, histogram_freq=1)
