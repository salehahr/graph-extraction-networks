from typing import Any, Iterable, Optional, Tuple

import numpy as np
import yaml
from pydantic import BaseModel, validator

from tools.data import get_skeletonised_ds


class Config(BaseModel):
    # user input in .yaml file
    img_length: int
    input_channels: int
    output_channels: int

    data_path: str
    save_path: str

    validation_fraction: float
    batch_size: int
    use_small_dataset: bool = False
    max_files: Optional[int] = None

    # generated from user input
    img_dims: Optional[Tuple[int, int]] = None

    dataset: Any = None
    test_ds: Any = None

    num_labels: Optional[int] = None
    num_validation: Optional[int] = None
    num_train: Optional[int] = None
    num_test: Optional[int] = None

    train_ids: Optional[Iterable] = None
    validation_ids: Optional[Iterable] = None

    steps_per_epoch: Optional[int]

    @validator("max_files")
    def check_max_files(cls, v, values):
        if values["use_small_dataset"] is True:
            return int(v)
        else:
            return None

    def __init__(self, filepath: str):
        with open(filepath) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        super(Config, self).__init__(**data)

        self.img_dims = (self.img_length, self.img_length)

        # training, validation, testing
        self.dataset = self.create_dataset()
        self.test_ds = self.create_dataset(is_test=True)

        self.num_labels = len(self.dataset)
        self.num_validation = int(self.validation_fraction * self.num_labels)
        self.num_train = self.num_labels - self.num_validation
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
            dataset = dataset.take(self.max_files)
        return dataset

    @property
    def training_ds(self):
        return self.dataset.take(self.num_train)

    @property
    def validation_ds(self):
        return self.dataset.skip(self.num_train)
