import math
import os
import shutil
import unittest
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

from model import VGG16
from model.unet import NodesNN, NodesNNExtended, UNet
from tools import (
    Config,
    EdgeExtractionDG,
    GraphExtractionDG,
    NodeExtractionDG,
    TestType,
    WandbConfig,
)
from tools.image import classify
from tools.plots import display_single_output, show_edge_predictions, show_predictions

cwd = os.getcwd()
assert "tests" in cwd

log_dir = os.path.join(cwd, "logs")


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
        cls.network = cls.config.network.node_extraction

        cls.wandb = WandbConfig("test_wandb_config.yaml")
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

    @unittest.skip("Skip wandb test.")
    def test_wandb_train(self):
        wandb.init(
            project=self.wandb.project,
            entity=self.wandb.entity,
            name=self.wandb.run_name,
            config=self.wandb.run_config,
        )
        self._wandb_train()
        wandb.finish()

    @unittest.skip("Skip parameter sweep.")
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

        cls.weights = os.path.join(cwd, "weights.hdf5")
        checkpoint_path = os.path.join(log_dir, "checkpoint_{epoch}.hdf5")

        cls.model = cls._init_model()
        cls.tensorboard = cls.model.tensorboard_callback(log_dir)
        cls.checkpoint = cls.model.checkpoint(checkpoint_path)

    @classmethod
    def _init_model(cls) -> UNet:
        unet = NodesNN(
            input_size=(*cls.config.img_dims, cls.network.input_channels),
            n_filters=64,
        )
        unet.build()

        return unet

    def _train(self):
        train_ds = NodeExtractionDG(self.config, self.network, TestType.TRAINING)
        val_ds = NodeExtractionDG(self.config, self.network, TestType.VALIDATION)

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
        wandb.init(
            project=self.wandb.project,
            entity=self.wandb.entity,
            name=self.wandb.run_name,
            config=self.wandb.run_config,
        )
        self.wandb_cb = self.model.wandb_callback()

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
        validation_ds = NodeExtractionDG(self.config, self.network, TestType.VALIDATION)
        show_predictions(self.model, validation_ds)

    def tearDown(self):
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


class TestEdgeNN(unittest.TestCase):
    """Only for training adjacencies."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        cls.network = cls.config.network.edge_extraction
        cls.weights = os.path.join(cwd, "weights_edge_nn.hdf5")

        num_filters = 4
        cls.model = cls._init_model(num_filters)

        cls.training_data, cls.validation_data = cls._get_data()

    @classmethod
    def _init_model(cls, num_filters: int) -> VGG16:
        edge_nn = VGG16(
            input_size=(*cls.config.img_dims, cls.network.input_channels),
            n_filters=num_filters,
            pretrained_weights=cls.weights,
        )
        edge_nn.build()
        return edge_nn

    @classmethod
    def _get_data(cls) -> Tuple[EdgeExtractionDG, EdgeExtractionDG]:
        orig_batch_size = cls.config.batch_size

        g_network = cls.config.network.graph_extraction
        cls.config.batch_size = 1
        graph_data = GraphExtractionDG(cls.config, g_network, TestType.TRAINING)
        cls.config.batch_size = orig_batch_size

        step_num = 0
        x_train, y_train = graph_data[step_num]
        x_val, y_val = graph_data[step_num + 1]

        training_data = EdgeExtractionDG(
            cls.config,
            cls.network,
            TestType.TRAINING,
            *x_train,
            y_train,
            with_path=False,
        )
        validation_data = EdgeExtractionDG(
            cls.config, cls.network, TestType.VALIDATION, *x_val, y_val, with_path=False
        )

        return training_data, validation_data

    def test_train(self) -> None:
        hist = self.model.fit(
            x=self.training_data,
            validation_data=self.validation_data,
            epochs=2,
            # steps per epoch
            steps_per_epoch=5,
            validation_steps=5,
        )

    def test_predict(self):
        """Visual test."""
        step_num = self._choose_step_num()
        print(f"Prediction at step {step_num}...")
        show_edge_predictions(self.model, self.validation_data, step_num)

    def _choose_step_num(self) -> int:
        """Ensures that a batch is chosen which contains a connected (adjacency) node pair."""
        has_adjacency = False
        step_num = 0

        while has_adjacency is False:
            _, adjacencies = self.validation_data[step_num]
            has_adjacency = 1 in adjacencies

            if has_adjacency:
                break
            else:
                step_num += 1

        return step_num
