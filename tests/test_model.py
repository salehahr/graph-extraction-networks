import math
import os
import shutil
import unittest
from typing import Optional

import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

from model.unet import NodesNN, NodesNNExtended, UNet
from tools import Config, DataGenerator, TestType, WandbConfig
from tools.image import classify
from tools.plots import display_single_output, show_predictions

cwd = os.getcwd()
assert "tests" in cwd

log_dir = os.path.join(cwd, "logs")


def _get_first_images(dataset, attribute: Optional[str] = None):
    """Returns the images of the first data point in the dataset."""
    file_id = 0
    batch_id = 0

    def get_skel_input(x):
        return np.expand_dims(x[batch_id], 0)

    if dataset.test_type == TestType.TRAINING:
        x, y = dataset[file_id]
        skel_input = get_skel_input(x)

        node_attributes = {
            "node_pos": np.expand_dims(y[0][batch_id], 0),
            "degrees": np.expand_dims(y[1][batch_id], 0),
            "node_types": np.expand_dims(y[2][batch_id], 0),
        }

        return skel_input, node_attributes[attribute]
    else:
        x = dataset[file_id]
        skel_input = get_skel_input(x)

        return skel_input


class TestSimpleModel(unittest.TestCase):
    """Tests a model with no branched outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        cls.network = cls.config.network.node_extraction

        cls.wandb = WandbConfig("test_wandb_config.yaml")
        cls.training_ds = DataGenerator(cls.config, cls.network, TestType.TRAINING)

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

    def _wandb_train(self):
        wandb_config = wandb.config
        model = self._get_simple_model(
            self.base_model.__dict__["node_pos"],
            "binary_crossentropy",
            optimiser=wandb_config.optimizer,
        )

        skel_input, y_gt = _get_first_images(self.training_ds, "node_pos")

        model.fit(
            x=skel_input,
            y=y_gt,
            callbacks=[WandbCallback(data_type="image")],
            epochs=wandb_config.epochs,
        )

    def test_wandb_train(self):
        wandb.init(
            project=self.wandb.project,
            entity=self.wandb.entity,
            name=self.wandb.run_name,
            config=self.wandb.run_config,
        )
        self._wandb_train()
        wandb.finish()

    def test_wandb_sweep(self):
        sweep_id = wandb.sweep(
            self.wandb.sweep_config,
            entity=self.wandb.entity,
            project=self.wandb.project,
        )
        wandb.agent(sweep_id, self._wandb_train, count=3)
        wandb.finish()

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
        cls.network = cls.config.network.node_extraction

        cls.wandb = WandbConfig("test_wandb_config.yaml")
        wandb.init(
            project=cls.wandb.project,
            entity=cls.wandb.entity,
            name=cls.wandb.run_name,
            config=cls.wandb.run_config,
        )

        cls.weights = os.path.join(cwd, "weights.hdf5")
        checkpoint_path = os.path.join(log_dir, "checkpoint_{epoch}.hdf5")

        cls.model = cls._init_model()
        cls.tensorboard = cls.model.tensorboard_callback(log_dir)
        cls.checkpoint = cls.model.checkpoint(checkpoint_path)
        cls.wandb_cb = cls.model.wandb_callback()

    @classmethod
    def _init_model(cls) -> UNet:
        unet = NodesNN(
            input_size=(*cls.config.img_dims, cls.network.input_channels),
            n_filters=64,
        )
        unet.build()

        return unet

    def _train(self):
        train_ds = DataGenerator(self.config, self.network, TestType.TRAINING)
        val_ds = DataGenerator(self.config, self.network, TestType.VALIDATION)

        hist = self.model.fit(
            x=train_ds,
            steps_per_epoch=len(train_ds),
            epochs=wandb.config.epochs,
            validation_data=val_ds,
            callbacks=[self.tensorboard, self.checkpoint, self.wandb_cb],
        )

        return hist.epoch, hist.history

    def setUp(self) -> None:
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)

    def test_train(self):
        """Smoke test to test that the model runs."""
        self._train()
        self.model.save_weights(self.weights)

    def test_no_nan_losses(self):
        """Tests that the losses are not NaN."""
        epochs, history = self._train()

        losses = history["loss"]
        for e in epochs:
            self.assertFalse(math.isnan(losses[e]))

    def test_predict(self):
        """Visual test."""
        self.config.batch_size = 2
        validation_ds = DataGenerator(self.config, self.network, TestType.VALIDATION)
        show_predictions(self.model, validation_ds)

    @classmethod
    def tearDownClass(cls):
        wandb.finish()


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
