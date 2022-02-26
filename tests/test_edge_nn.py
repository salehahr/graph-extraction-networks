import os
import shutil
import unittest

from model import VGG16
from tools import TestType, get_eedg, get_gedg, run


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
        cls.checkpoint = cls.model.checkpoint(cls.config.checkpoint_path)

        cls.graph_data = get_gedg(cls.config)
        cls.edge_data = get_eedg(cls.config, cls.run_config, cls.graph_data)

    @classmethod
    def _init_model(cls, num_filters: int) -> VGG16:
        edge_nn = VGG16(
            input_size=(*cls.config.img_dims, cls.network.input_channels),
            n_filters=num_filters,
            pretrained_weights=cls.weights,
        )
        edge_nn.build()
        return edge_nn

    def setUp(self) -> None:
        directories = [self.config.checkpoint_dir]
        for d in directories:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
        run.start(self.run_config)

    def test_train(self) -> None:
        short_training_run = False
        run.train(
            self.model, self.edge_data, debug=short_training_run, predict_frequency=2
        )

    def test_predict(self) -> None:
        run.predict(
            self.model, self.edge_data[TestType.VALIDATION], max_pred=5, show=True
        )

    def tearDown(self) -> None:
        run.end()
