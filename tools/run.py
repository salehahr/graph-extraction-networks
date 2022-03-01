from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union

import numpy as np
import wandb
from keras.callbacks import Callback
from wandb.integration.keras import WandbCallback

from model import EdgeNN
from tools import Config, RunConfig
from tools.plots import plot_node_pairs_on_skel
from tools.postprocessing import classify
from tools.TestType import TestType

if TYPE_CHECKING:
    import tensorflow as tf

    from model import UNet
    from tools import EdgeDG


def get_configs(config_fp: str, run_config_fp: str) -> Tuple[Config, RunConfig]:
    data_config = Config(config_fp)
    run_config = RunConfig(run_config_fp, data_config=data_config)
    return data_config, run_config


def start(
    run_config: RunConfig,
    run_name: Optional[str] = None,
    run_id: Optional[str] = None,
    is_sweep: bool = False,
) -> wandb.run:
    run_params = None if is_sweep else run_config.parameters

    if not run_name and not is_sweep:
        run_name = run_config.run_name

    resume = "must" if run_config.resume is True else run_config.resume
    reinit = True if run_config.resume is True else run_config.resume
    id_ = run_id if run_id is not None else run_config.run_id

    run_ = wandb.init(
        project=run_config.project,
        entity=run_config.entity,
        name=run_name,
        config=run_params,
        resume=resume,
        reinit=reinit,
        id=id_,
    )
    return run_


def sweep(run_config: RunConfig, train_func: Callable, count: int):
    # configure the sweep
    sweep_id = wandb.sweep(
        run_config.sweep_config,
        entity=run_config.entity,
        project=run_config.project,
    )

    # run the sweep
    wandb.agent(sweep_id, train_func, count=count)


def load_model(
    config: Config,
    run_config: RunConfig,
    model_: Optional[EdgeNN] = None,
) -> EdgeNN:
    """Either initialises a new model or loads an existing model."""
    # initialise model
    if model_ is None:
        model_ = EdgeNN(
            input_size=(*config.img_dims, 1),
            n_filters=wandb.config.n_filters,
            n_conv2_blocks=wandb.config.n_conv2_blocks,
            n_conv3_blocks=wandb.config.n_conv3_blocks,
            pretrained_weights=run_config.weights_path,
            learning_rate=run_config.parameters.learning_rate,
        )
        model_.build()

    # load weights on resumed run
    if wandb.run.resumed:
        try:
            best_model = wandb.restore(
                "model-best.h5",
                run_path=f"{run_config.entity}/{run_config.project}/{run_config.run_id}",
            )
            model_.load_weights(best_model.name)
            model_.recompile()
        except ValueError as ve:
            print(ve)
            print(
                "Warning: model not found on resumed run -- using a newly initialised model."
            )
        except Exception as e:
            print(e)
            print("Load model failed.")
            sys.exit(1)

    return model_


def train(
    model_: Union[EdgeNN, UNet],
    data: Dict[TestType, EdgeDG],
    epochs: Optional[int] = None,
    max_num_images: Optional[int] = None,
    steps_in_epoch: Optional[int] = None,
    test_type: TestType = TestType.TRAINING,
    predict_frequency: int = 10,
    debug: bool = False,
) -> tf.keras.callbacks.History:
    epochs = wandb.config.epochs if epochs is None else epochs

    validation_data = (
        data[TestType.VALIDATION] if test_type is TestType.TRAINING else None
    )

    if debug is True:
        num_steps = 3
    else:
        num_steps = steps_in_epoch

        if max_num_images is not None:
            num_steps = math.ceil(
                max_num_images / data[TestType.TRAINING].images_in_batch
            )

    history = model_.fit(
        x=data[test_type],
        validation_data=validation_data,
        initial_epoch=wandb.run.step,
        epochs=epochs,
        steps_per_epoch=num_steps,
        validation_steps=num_steps,
        callbacks=[
            WandbCallback(save_weights_only=True),
            PredictionCallback(predict_frequency, validation_data),
        ],
    )

    return history


def save(model_: EdgeNN, filename: str, in_wandb_dir: bool = True) -> str:
    if in_wandb_dir:
        run_dir = wandb.run.dir
        filename = filename if run_dir in filename else os.path.join(run_dir, filename)
        model_.save_weights(filename)
    else:
        model_.save_weights(filename)
    return wandb.run.id


def predict(
    model_: EdgeNN,
    val_data: EdgeDG,
    max_pred: int = 5,
    only_adj_nodes: bool = False,
    show: bool = False,
    description: Optional[str] = None,
):
    prediction = wandb.Artifact(
        f"run_{wandb.run.id}", type="predictions", description=description
    )
    table = wandb.Table(columns=["adj_pred", "adj_true", "image"])

    num_imgs = val_data.images_in_batch
    num_combos = val_data.node_pairs_image

    step_num = 0
    id_in_batch = 0
    max_steps = math.ceil(max_pred / val_data.node_pairs_batch)

    for step in range(max_steps):
        if only_adj_nodes:
            # Only show predictions for adjacent nodes
            step_num, id_in_batch = choose_step_num(
                val_data, step_num=step_num, id_in_batch=id_in_batch, pick_adj=True
            )
        else:
            step_num = step

        combo_img, adj_true = val_data[step_num]
        adj_pred, _ = classify(model_.predict(combo_img))

        for i in range(num_imgs):
            idx = i * num_combos

            skel_img = np.float32(combo_img[idx, ..., 0].numpy())

            for ii in range(num_combos):
                node_pair_img = np.float32(combo_img[idx, ..., 1].numpy())
                pair_xy = [p for p in np.fliplr(np.argwhere(node_pair_img))]

                rgb_img = plot_node_pairs_on_skel(skel_img, [pair_xy], show=show)
                table.add_data(
                    int(adj_pred[idx]), int(adj_true[idx]), wandb.Image(rgb_img)
                )

                idx += 1
                if idx >= max_pred:
                    break

            if idx >= max_pred:
                break

        step_num, id_in_batch = increment_step(
            step_num, id_in_batch, val_data.batch_size
        )

    prediction.add(table, "predictions")

    wandb.log_artifact(prediction)


def choose_step_num(
    val_data: EdgeDG,
    step_num: int = 0,
    id_in_batch: int = 0,
    pick_adj: bool = True,
) -> Tuple[int, int]:
    """Ensures that a batch is chosen which contains a connected (adjacency) node pair."""
    found = False

    while found is False:
        _, adjacencies = val_data[step_num]
        adj = adjacencies[id_in_batch]
        found = adj.numpy().squeeze() == 1 if pick_adj is True else (0 in adjacencies)

        if found:
            break

        step_num, id_in_batch = increment_step(
            step_num, id_in_batch, val_data.batch_size
        )

    return step_num, id_in_batch


def increment_step(step_num: int, id_in_batch: int, batch_size: int) -> Tuple[int, int]:
    use_next_batch = id_in_batch >= (batch_size - 1)

    if use_next_batch:
        step_num += 1
        id_in_batch = 0
    else:
        # step_num = step_num -- remains the same
        id_in_batch += 1

    return step_num, id_in_batch


def end() -> None:
    wandb.finish()


class PredictionCallback(Callback):
    def __init__(
        self,
        frequency: int,
        validation_data: EdgeDG,
        total_epochs: Optional[int] = None,
    ):
        super().__init__()

        self.frequency = frequency
        self.validation_data = validation_data
        self.total_epochs = (
            total_epochs if total_epochs is not None else wandb.config.epochs
        )

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            predict(
                self.model,
                self.validation_data,
                description=f"Prediction on epoch {epoch + 1}/{self.total_epochs}.",
            )
