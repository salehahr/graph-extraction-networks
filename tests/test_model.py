import math
import os
import shutil
import unittest

from model.unet import UNet
from tools import Config, DataGenerator, TestType

cwd = os.getcwd()
assert "tests" in cwd

log_dir = os.path.join(cwd, "logs")


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        cls.model = cls._init_model()
        cls.tensorboard = cls._init_tensorboard()
        cls.weights = os.path.join(cwd, "weights.hdf5")

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

    def test_predict(self):
        validation_ds = DataGenerator(self.config, TestType.VALIDATION)

        self.model.load_weights(self.weights)

        results = self.model.predict(x=validation_ds, verbose=1)

        return
