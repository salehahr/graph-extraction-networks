import math
import os
import shutil
import unittest

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model.unet import UNet
from tools import Config, DataGenerator, TestType

cwd = os.getcwd()
assert "tests" in cwd

log_dir = os.path.join(cwd, "logs")


class TestSimpleModel(unittest.TestCase):
    """Tests a model with no branched outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        cls.training_ds = DataGenerator(cls.config, TestType.TRAINING)

        cls.base_model = cls._init_model()
        cls.input_layer = cls.base_model.get_layer("input").output

    @classmethod
    def _init_model(cls) -> UNet:
        unet = UNet(
            input_size=(*cls.config.img_dims, cls.config.input_channels),
            n_filters=64,
        )
        unet.build()

        return unet

    def _get_simple_model(self, output, loss):
        """Appends a single output to the UNet model."""
        model = tf.keras.Model(inputs=self.input_layer, outputs=output)
        model.compile(optimizer="Adam", loss=loss, metrics=["accuracy"])
        return model

    def _test_simple_model(self, output: str, loss: str):
        model = self._get_simple_model(self.base_model.__dict__[output], loss)

        skel_input, y_gt = self._get_first_images(self.training_ds, output)

        hist = model.fit(x=skel_input, y=y_gt)
        losses = hist.history["loss"]

        # make sure losses are not nan
        for l in losses:
            self.assertFalse(math.isnan(l))

        y_pred = model.predict(x=skel_input, verbose=1)
        y_pred = (
            self.binary_classify(y_pred)
            if output == "node_pos"
            else self.categorical_classify(y_pred)
        )

        self.display([skel_input[0], y_gt[0], y_pred[0]], output)

    def test_node_pos_model(self):
        self._test_simple_model("node_pos", "binary_crossentropy")

    def test_node_degrees_model(self):
        self._test_simple_model("degrees", "sparse_categorical_crossentropy")

    def test_node_types_model(self):
        self._test_simple_model("node_types", "sparse_categorical_crossentropy")

    @staticmethod
    def _get_first_images(dataset, attribute: str):
        file_id = 0
        x, y = dataset[file_id]

        batch_id = 0
        skel_input = np.expand_dims(x[batch_id], 0)
        node_attributes = {
            "node_pos": np.expand_dims(y[0][batch_id], 0),
            "degrees": np.expand_dims(y[1][batch_id], 0),
            "node_types": np.expand_dims(y[2][batch_id], 0),
        }

        return skel_input, node_attributes[attribute]

    @staticmethod
    def display(display_list: list, big_title: str):
        title = ["Input Image", "True Mask", "Predicted Mask"]

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis("off")

        plt.suptitle(big_title)
        plt.show()

    @staticmethod
    def binary_classify(pred_mask):
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask.astype(np.uint8)

    @staticmethod
    def categorical_classify(pred_mask) -> np.ndarray:
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask.numpy()


class TestUntrainedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        cls.weights = os.path.join(cwd, "weights.hdf5")
        cls.model = cls._init_model()
        cls.tensorboard = cls._init_tensorboard()

    @classmethod
    def _init_model(cls) -> UNet:
        unet = UNet(
            input_size=(*cls.config.img_dims, cls.config.input_channels),
            n_filters=64,
        )
        unet.build()

        return unet

    @classmethod
    def _init_tensorboard(cls):
        return cls.model.tensorboard_callback(log_dir)

    def _train(self, num_epochs: int = 2):
        train_ds = DataGenerator(self.config, TestType.TRAINING)

        hist = self.model.fit(
            x=train_ds,
            steps_per_epoch=len(train_ds),
            epochs=num_epochs,
            callbacks=[self.tensorboard],
        )

        return hist.epoch, hist.history

    def setUp(self) -> None:
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)

    def test_train(self):
        """Smoke test to test that the model runs."""
        num_epochs = 2
        self._train(num_epochs)
        self.model.save_weights(self.weights)

    def test_no_nan_losses(self):
        """Tests that the losses are not NaN."""
        epochs, history = self._train()

        losses = history["loss"]
        for e in epochs:
            self.assertFalse(math.isnan(losses[e]))


class TestTrainedModel(TestUntrainedModel):
    @classmethod
    def _init_model(cls) -> UNet:
        unet = UNet(
            input_size=(*cls.config.img_dims, cls.config.input_channels),
            n_filters=64,
            pretrained_weights=cls.weights,
        )
        unet.build()

        return unet

    def test_predict(self):
        validation_ds = DataGenerator(self.config, TestType.VALIDATION)

        self.model.load_weights(self.weights)

        results = self.model.predict(x=validation_ds, verbose=1)

        batch_id = 0
        node_pos = results[0][batch_id]
        degrees = results[1][batch_id]
        node_types = results[2][batch_id]

        return
