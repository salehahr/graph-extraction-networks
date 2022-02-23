import os
import shutil
import unittest

from model import VGG16
from tools import Config, RunConfig, TestType, get_eedg, get_gedg, run
from tools.plots import show_edge_predictions


class TestEdgeNN(unittest.TestCase):
    """Only for training adjacencies."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        cls.run_config = RunConfig(
            "test_wandb_config_edge.yaml", data_config=cls.config
        )
        cls.network = cls.config.network.edge_extraction

        num_filters = 4
        cls.weights = os.path.join(cls.config.base_path, "weights_edge_nn.hdf5")
        cls.model = cls._init_model(num_filters)
        cls.checkpoint = cls.model.checkpoint(cls.config.checkpoint_path)

        cls.graph_data = get_gedg(cls.config, batch_size=1)
        edge_data = get_eedg(
            cls.config, cls.run_config.node_pairs_in_batch, cls.graph_data
        )
        cls.training_data = edge_data[TestType.TRAINING]
        cls.validation_data = edge_data[TestType.VALIDATION]

    def setUp(self) -> None:
        directories = [self.config.checkpoint_dir]
        for d in directories:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)

    @classmethod
    def _init_model(cls, num_filters: int) -> VGG16:
        edge_nn = VGG16(
            input_size=(*cls.config.img_dims, cls.network.input_channels),
            n_filters=num_filters,
            pretrained_weights=cls.weights,
        )
        edge_nn.build()
        return edge_nn

    def test_train(self) -> None:
        run.start(self.run_config)

        data = get_eedg(
            self.config, self.run_config.node_pairs_in_batch, self.graph_data
        )
        run.train(self.model, data, debug=True)
        run.predict(self.model, self.validation_data, max_pred=3, alternate=True)

        run.end()

    def test_train_multiple(self):
        """Train over multiple images."""
        num_images = 3
        old_run_id = None
        run_name = "test_edge_multiple"
        model_ = self.model

        for i in range(num_images):
            do_resume = False if i == 0 else "must"
            run.start(
                self.run_config,
                resume=do_resume,
                reinit=True,
                _id=old_run_id,
                run_name=run_name,
            )

            # generate node combinations
            edge_data = get_eedg(
                self.config,
                self.run_config.node_pairs_in_batch,
                self.graph_data,
                step_num=i,
            )

            # init model/reload model on resumed run
            model_ = run.load_model(self.config, self.run_config, model_=model_)

            # train and save for next run
            run.train(model_, edge_data, debug=True)
            old_run_id = run.save(model_, self.run_config.model_filename)

            run.end()

    def test_predict(self):
        """Visual test."""
        step_num = run.choose_step_num(self.validation_data)
        print(f"Prediction at step {step_num}...")
        show_edge_predictions(self.model, self.validation_data, step_num)
