import os
import shutil
import unittest

from model import EdgeNN
from tools import TestType, get_eedg, run


class TestEdgeNN(unittest.TestCase):
    """Only for training adjacencies."""

    @classmethod
    def setUpClass(cls) -> None:
        data_config_fp = "test_config.yaml"
        run_config_fp = "test_wandb_config_edge.yaml"
        cls.config, cls.run_config = run.get_configs(data_config_fp, run_config_fp)
        cls.network = cls.config.network.edge_extraction

        num_filters = 4
        cls.weights = os.path.join(cls.config.base_path, "weights_edge_nn.hdf5")
        cls.model = cls._init_model(num_filters)

        cls.edge_data = get_eedg(cls.config, cls.run_config)

    @classmethod
    def _init_model(cls, num_filters: int) -> EdgeNN:
        edge_nn = EdgeNN(
            input_size=(*cls.config.img_dims, 1),
            n_filters=num_filters,
            batch_norm=cls.run_config.parameters.batch_norm,
            pretrained_weights=None,
        )
        edge_nn.build()
        return edge_nn

    def setUp(self) -> None:
        directories = [self.config.checkpoint_dir]
        for d in directories:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)

    def test_model_output_shapes(self):
        sum_output = self.model.get_layer("summation").output
        self.assertEqual(sum_output.shape[1:], (256, 256, 1))

    def test_train(self) -> None:
        run.start(self.run_config)
        short_training_run = False
        run.train(
            self.model, self.edge_data, debug=short_training_run, predict_frequency=2
        )
        run.end()

    def test_predict(self) -> None:
        run.start(self.run_config)
        run.predict(
            self.model, self.edge_data[TestType.VALIDATION], max_pred=5, show=True
        )
        run.end()

    def test_overfit(self) -> None:
        """Sanity check: model trained on only a small amount of data SHUOLD overfit."""
        data_config_fp = "small_ds.yaml"
        run_config_fp = "edge_overfit.yaml"
        validate = True
        debug = False

        config, run_config = run.get_configs(data_config_fp, run_config_fp)

        edge_data = get_eedg(config, run_config, validate=validate)

        run.start(run_config)
        run.train(
            self.model, edge_data, validate=validate, debug=debug, predict_frequency=2
        )
        run.end()

    def test_sweep(self):
        sweep_id = None
        run.sweep(self.run_config, self._train_for_sweep, count=5, sweep_id=sweep_id)

    def _train_for_sweep(self):
        run.start(self.run_config, is_sweep=True)
        short_training_run = True

        model = run.load_model(self.config, self.run_config, do_sweep=True)

        run.train(model, self.edge_data, debug=short_training_run, predict_frequency=2)
        run.end()
