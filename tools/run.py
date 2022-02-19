from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import wandb
from wandb.integration.keras import WandbCallback

from model import VGG16
from tools import Config, RunConfig
from tools.plots import plot_node_pairs_on_skel
from tools.postprocessing import eedg_coordinates, eedg_predict
from tools.TestType import TestType

if TYPE_CHECKING:
    import tensorflow as tf

    from model import UNet
    from tools import EdgeDGSingle


def get_configs(config_fp: str, run_name: str) -> Tuple[Config, RunConfig]:
    data_config = Config(config_fp)
    run_config_fp = os.path.join(data_config.base_path, f"configs/{run_name}.yaml")
    run_config = RunConfig(run_config_fp, run_name, data_config=data_config)
    return data_config, run_config


def check_weights(run_config: RunConfig) -> None:
    # moot because the way the model is saved has changed
    weights_exist = os.path.isfile(run_config.weights_path)

    if not run_config.resume:
        assert weights_exist is False
    # else:
    #     assert weights_exist is True


def start(
    run_config: RunConfig,
    resume: Union[bool, str] = False,
    reinit: bool = False,
    _id: Optional[str] = None,
    run_name: Optional[str] = None,
    is_sweep: bool = False,
) -> wandb.run:
    run_params = None if is_sweep else run_config.parameters

    if not run_name and not is_sweep:
        run_name = run_config.run_name

    run_ = wandb.init(
        project=run_config.project,
        entity=run_config.entity,
        name=run_name,
        config=run_params,
        resume=resume,
        reinit=reinit,
        id=_id,
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
    model_: Optional[VGG16] = None,
) -> VGG16:
    """Either initialises a new model or loads an existing model."""
    # initialise model
    if model_ is None:
        model_ = VGG16(
            input_size=(
                *config.img_dims,
                config.network.edge_extraction.input_channels,
            ),
            n_filters=wandb.config.n_filters,
            n_conv2_blocks=wandb.config.n_conv2_blocks,
            n_conv3_blocks=wandb.config.n_conv3_blocks,
            pretrained_weights=run_config.weights_path,
        )
        model_.build()

    # load weights on resumed run
    if wandb.run.resumed:
        h5_model = wandb.restore(run_config.model_filename)
        model_.load_weights(h5_model.name)
        model_.recompile()

    return model_


def train(
    model_: Union[VGG16, UNet],
    data: Dict[TestType, Any],
    epochs: Optional[int] = None,
    test_type: TestType = TestType.TRAINING,
    debug: bool = False,
) -> tf.keras.callbacks.History:
    epochs = wandb.config.epochs if epochs is None else epochs

    validation_data = (
        data[TestType.VALIDATION] if test_type is TestType.TRAINING else None
    )

    num_steps = 3 if debug is True else None

    history = model_.fit(
        x=data[test_type],
        validation_data=validation_data,
        initial_epoch=wandb.run.step,
        epochs=epochs + wandb.run.step,
        steps_per_epoch=num_steps,
        validation_steps=num_steps,
        callbacks=[WandbCallback(save_weights_only=True)],
    )

    return history


def save(model_: VGG16, filename: str, in_wandb_dir: bool = True) -> str:
    if in_wandb_dir:
        run_dir = wandb.run.dir
        filename = filename if run_dir in filename else os.path.join(run_dir, filename)
        model_.save_weights(filename)
    else:
        model_.save_weights(filename)
    return wandb.run.id


def predict(
    model_: VGG16,
    val_data: EdgeDGSingle,
    max_pred: int = 3,
    alternate: bool = False,
    only_adj_nodes: bool = True,
):
    prediction = wandb.Artifact(f"run_{wandb.run.id}", type="predictions")
    table = wandb.Table(columns=["adj_pred", "adj_true", "image"])

    orig_batch_size = val_data.batch_size
    val_data.batch_size = 1

    step_num = 0
    for i in range(max_pred):
        if alternate:
            # Alternate between positive and negative adjacencies
            pick_adj = True if (i % 2 == 0) else False
            step_num = choose_step_num(val_data, step_num=step_num, pick_adj=pick_adj)
        else:
            if not only_adj_nodes:
                step_num = i
            else:
                # Only show predictions for adjacent nodes
                step_num = choose_step_num(val_data, step_num=step_num, pick_adj=True)

        adj_true, adj_pred = eedg_predict(val_data, model_, step_num)
        pairs_xy = eedg_coordinates(val_data, step_num)

        rgb_img = plot_node_pairs_on_skel(val_data.skel_img, pairs_xy)

        table.add_data(int(adj_pred), int(adj_true), wandb.Image(rgb_img))

        step_num += 1

    val_data.batch_size = orig_batch_size
    prediction.add(table, "predictions")

    wandb.log_artifact(prediction)


def choose_step_num(
    val_data: EdgeDGSingle, step_num: int = 0, pick_adj: bool = True
) -> int:
    """Ensures that a batch is chosen which contains a connected (adjacency) node pair."""
    found = False

    while found is False:
        _, adjacencies = val_data[step_num]
        found = (1 in adjacencies) if pick_adj is True else (0 in adjacencies)

        if found:
            break
        else:
            step_num += 1

    return step_num


def end() -> None:
    wandb.finish()
