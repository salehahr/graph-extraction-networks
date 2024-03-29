import math
import os
import shutil
import unittest
from typing import Optional

import numpy as np
import tensorflow as tf
from wandb.keras import WandbCallback

import tools.run as run
from model.unet import NodesNN, NodesNNExtended, UNet
from tools import Config, NodeExtractionDG, RunConfig, TestType
from tools.plots import display_single_output, show_predictions
from tools.postprocessing import classify

assert "tests" in os.getcwd()


def _get_first_images(data_generator, attribute: Optional[str] = None):
    """Returns the images of the first data point in the dataset."""
    file_id = 0
    batch_id = 0

    x_batch, y_batch = data_generator[file_id]

    x = x_batch[batch_id].numpy()
    skel_input = np.expand_dims(x, 0)

    node_attributes = {
        "node_pos": np.expand_dims(y_batch[0][batch_id].numpy(), 0),
        "degrees": np.expand_dims(y_batch[1][batch_id].numpy(), 0),
        "node_types": np.expand_dims(y_batch[2][batch_id].numpy(), 0),
    }

    return skel_input, node_attributes[attribute]


class TestSimpleModel(unittest.TestCase):
    """Tests a model with no branched outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        cls.run_config = RunConfig("test_wandb_config.yaml", data_config=cls.config)
        cls.network = cls.config.network.node_extraction

        cls.training_ds = NodeExtractionDG(cls.config, cls.network, TestType.TRAINING)

        cls.base_model = cls._init_model()
        cls.input_layer = cls.base_model.get_layer("input").output

    @classmethod
    def _init_model(cls) -> UNet:
        unet = NodesNNExtended(
            input_size=(*cls.config.img_dims, cls.network.input_channels),
            n_filters=4,
            depth=5,
        )
        unet.build()

        return unet

    def _get_simple_model(self, output, loss, optimiser="Adam"):
        """Appends a single output to the UNet model."""
        model = tf.keras.Model(inputs=self.input_layer, outputs=output)
        model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])
        return model

    def _test_simple_model(self, output: str, loss: str):
        model = self._get_simple_model(self.base_model.__dict__[output], loss)

        skel_input, y_gt = _get_first_images(self.training_ds, output)

        hist = model.fit(x=skel_input, y=y_gt)
        losses = hist.history["loss"]

        # make sure losses are not nan
        for l in losses:
            self.assertFalse(math.isnan(l))

        y_pred = model.predict(x=skel_input, verbose=1)
        y_pred, is_binary = classify(y_pred)

        display_single_output([skel_input[0], y_gt[0], y_pred[0]], output)

        return is_binary

    def _train_for_sweep(self):
        run_ = run.start(self.run_config, is_sweep=True)

        model = self._get_simple_model(
            self.base_model.__dict__["node_pos"],
            "binary_crossentropy",
            optimiser=run_.config.optimizer,
        )

        skel_input, y_gt = _get_first_images(self.training_ds, "node_pos")

        model.fit(
            x=skel_input,
            y=y_gt,
            callbacks=[WandbCallback()],
            epochs=self.run_config.parameters["epochs"],
            steps_per_epoch=3,
        )

    def test_wandb_sweep(self):
        run.sweep(self.run_config, self._train_for_sweep, count=2)

    def test_node_pos_model(self):
        is_binary = self._test_simple_model("node_pos", "binary_crossentropy")
        self.assertTrue(is_binary)

    def test_node_degrees_model(self):
        is_binary = self._test_simple_model(
            "degrees", "sparse_categorical_crossentropy"
        )
        self.assertFalse(is_binary)

    def test_node_types_model(self):
        is_binary = self._test_simple_model(
            "node_types", "sparse_categorical_crossentropy"
        )
        self.assertFalse(is_binary)


class TestUntrainedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        cls.run_config = RunConfig("test_wandb_config.yaml", data_config=cls.config)
        cls.network = cls.config.network.node_extraction

        cls.weights = os.path.join(cls.config.base_path, "weights.hdf5")
        cls.model = cls._init_model()
        cls.tensorboard = cls.model.tensorboard_callback(cls.config.log_path)
        cls.checkpoint = cls.model.checkpoint(cls.config.checkpoint_path)

    @classmethod
    def _init_model(cls) -> UNet:
        unet = NodesNN(
            input_size=(*cls.config.img_dims, cls.network.input_channels),
            n_filters=64,
        )
        unet.build()

        return unet

    def _train(self):
        data = {
            test: NodeExtractionDG(self.config, self.network, test) for test in TestType
        }
        history = run.train(self.model, data, debug=True)

        return history

    def setUp(self) -> None:
        if os.path.isdir(self.config.log_path):
            shutil.rmtree(self.config.log_path)
        run.start(self.run_config)

    def test_train(self):
        """Smoke test to test that the model runs."""
        self._train()
        run.save(self.run_config.weights_path, in_wandb_dir=False)

    def test_no_nan_losses(self):
        """Tests that the losses are not NaN."""
        history = self._train()
        losses = history.history["loss"]

        for e in history.epoch:
            self.assertFalse(math.isnan(losses[e]))

    def test_predict(self):
        """Visual test."""
        self.config.batch_size = 2
        validation_ds = NodeExtractionDG(self.config, self.network, TestType.VALIDATION)
        show_predictions(self.model, validation_ds)

    def tearDown(self):
        run.end()


class TestTrainedModel(TestUntrainedModel):
    @classmethod
    def _init_model(cls) -> UNet:
        unet = NodesNN(
            input_size=(*cls.config.img_dims, cls.network.input_channels),
            n_filters=64,
            pretrained_weights=cls.weights,
        )
        unet.build()

        return unet
