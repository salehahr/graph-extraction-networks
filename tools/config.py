from __future__ import annotations

import os
from typing import Any, Iterable, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, validator

from tools.data import get_skeletonised_ds
from tools.NetworkType import NetworkType


class InputConfig(BaseModel):
    # user input in .yaml file
    id: Union[int, NetworkType]
    input_channels: int
    output_channels: int

    @validator("id")
    def set_id(cls, v) -> NetworkType:
        return NetworkType(v)

    def __init__(self, data):
        super(InputConfig, self).__init__(**data)


class NNConfig(BaseModel):
    # user input in .yaml file
    node_extraction: Union[dict, InputConfig]
    graph_extraction: Union[dict, InputConfig]
    edge_extraction: Union[dict, InputConfig]

    @validator("node_extraction", "graph_extraction", "edge_extraction")
    def set_network(cls, v):
        return InputConfig(v)

    def __init__(self, data):
        super(NNConfig, self).__init__(**data)


class Config(BaseModel):
    # user input in .yaml file
    img_length: int
    network: Union[dict, NNConfig]

    base_path: str
    data_path: str
    save_path: str

    validation_fraction: float
    batch_size: int
    node_pairs_in_batch: int
    use_small_dataset: bool = False
    max_files: Optional[int] = None

    # generated from user input
    img_dims: Optional[Tuple[int, int]] = None

    log_path: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    checkpoint_path: Optional[str] = None

    dataset: Any = None
    test_ds: Any = None

    num_labels: Optional[int] = None
    num_validation: Optional[int] = None
    num_train: Optional[int] = None
    num_test: Optional[int] = None

    train_ids: Optional[Iterable] = None
    validation_ids: Optional[Iterable] = None

    run: Optional[WandbConfig] = None
    steps_per_epoch: Optional[int] = None

    @validator("max_files")
    def check_max_files(cls, v, values):
        if values["use_small_dataset"] is True:
            return int(v)
        else:
            return None

    @validator("network")
    def set_network(cls, v):
        return NNConfig(v)

    def __init__(self, filepath: str):
        with open(filepath) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        super(Config, self).__init__(**data)

        self.img_dims = (self.img_length, self.img_length)

        # paths
        self.log_path = os.path.join(self.base_path, "logs")
        self.checkpoint_dir = os.path.join(self.base_path, "checkpoints")
        self.checkpoint_path = os.path.join(
            self.checkpoint_dir, "checkpoint_{epoch}.hdf5"
        )

        # training, validation, testing
        self.dataset = self.create_dataset()
        self.num_labels = len(self.dataset)
        self.num_validation = int(self.validation_fraction * self.num_labels)
        self.num_train = self.num_labels - self.num_validation

        self.test_ds = self.create_dataset(is_test=True)
        self.num_test = len(self.test_ds)

        print(
            f"Total: {self.num_labels} training data --",
            f"[{self.num_train} training]",
            f"[{self.num_validation} validation]",
            f"[{self.num_test} test]",
        )

    def create_dataset(self, is_test=False):
        dataset = get_skeletonised_ds(self.data_path, seed=13, is_test=is_test)
        if self.use_small_dataset:
            max_files = self.num_validation if is_test else self.max_files
            dataset = dataset.take(max_files)
        return dataset

    @property
    def training_ds(self):
        return self.dataset.take(self.num_train)

    @property
    def validation_ds(self):
        return self.dataset.skip(self.num_train)


class WandbConfig(BaseModel):
    # user input in .yaml file
    project: str
    entity: str
    run_name: Optional[str]
    run_config: dict
    sweep_config: Optional[dict]

    # derived
    ds_config: Optional[Config] = None
    weights_path: Optional[str] = None

    def __init__(
        self, filepath: str, name: str = None, ds_config: Optional[Config] = None
    ):
        with open(filepath) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        super(WandbConfig, self).__init__(**data)

        self.run_name = name if name is not None else self.run_name

        if ds_config:
            self.ds_config = ds_config
            ds_config.run = self

            self.weights_path = os.path.join(
                ds_config.base_path, f"weights_{self.run_name}.hdf5"
            )
