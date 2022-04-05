from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import wandb
from keras.callbacks import Callback, ModelCheckpoint
from wandb.integration.keras import WandbCallback

from model import EdgeNN
from tools import Config, RunConfig
from tools.plots import plot_node_pairs_on_skel
from tools.postprocessing import classify, smooth
from tools.TestType import TestType

if TYPE_CHECKING:
    import tensorflow as tf

    from model import UNet
    from tools import EdgeDG


def get_configs(
    config_fp: str, run_config_fps: Union[List[str], str]
) -> Tuple[Config, Union[RunConfig, List[RunConfig]]]:
    data_config = Config(config_fp)

    if isinstance(run_config_fps, str):
        run_config = RunConfig(run_config_fps, data_config=data_config)
    else:
        run_config = [RunConfig(fp, data_config=data_config) for fp in run_config_fps]

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
    # wandb.run.define_metric("val_precision", summary="best", goal="maximize")

    best_model_fp = os.path.join(wandb.run.dir, "model-best.h5")
    wandb.save(glob_str=best_model_fp, base_path=wandb.run.dir, policy="live")

    return run_


def sweep(
    run_config: RunConfig,
    train_func: Callable,
    count: Optional[int] = None,
    sweep_id: Optional[str] = None,
):
    # configure the sweep
    sweep_id = (
        wandb.sweep(
            run_config.sweep_config,
            entity=run_config.entity,
            project=run_config.project,
        )
        if sweep_id is None
        else sweep_id
    )

    # run the sweep
    wandb.agent(
        sweep_id,
        train_func,
        count=count,
        entity=run_config.entity,
        project=run_config.project,
    )


def load_model(
    config: Config,
    run_config: RunConfig,
    model_: Optional[EdgeNN] = None,
    do_sweep: bool = False,
    do_train: bool = True,
) -> EdgeNN:
    """Either initialises a new model or loads an existing model."""
    # initialise model
    params = wandb.config if do_sweep else run_config.parameters

    if model_ is None:
        model_ = EdgeNN(
            input_size=(*config.img_dims, 1),
            n_filters=params.n_filters,
            batch_norm=params.batch_norm,
            n_conv2_blocks=params.n_conv2_blocks,
            n_conv3_blocks=params.n_conv3_blocks,
            pretrained_weights=run_config.pretrained_weights,
            learning_rate=params.learning_rate,
            optimiser=params.optimiser,
        )
        model_.build()

    # load weights on resumed run
    if do_train is True and wandb.run.resumed is True:
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
    validate: bool = True,
    epochs: Optional[int] = None,
    max_num_images: Optional[int] = None,
    steps_in_epoch: Optional[int] = None,
    predict_frequency: int = 10,
    debug: bool = False,
) -> tf.keras.callbacks.History:
    epochs = wandb.config.epochs if epochs is None else epochs

    training_data = data[TestType.TRAINING]
    validation_data = data[TestType.VALIDATION] if validate is True else None

    # calculate number of steps per epoch
    num_steps_train = _get_num_steps(
        training_data, steps_in_epoch, max_num_images, debug
    )
    max_num_images_val = max_num_images // 2 if max_num_images else None
    num_steps_validation = _get_num_steps(
        validation_data, steps_in_epoch, max_num_images_val, debug
    )

    history = model_.fit(
        x=training_data,
        validation_data=validation_data,
        initial_epoch=wandb.run.step,
        epochs=epochs,
        steps_per_epoch=num_steps_train,
        validation_steps=num_steps_validation,
        callbacks=[
            WandbCallback(save_weights_only=True),
            PredictionCallback(predict_frequency, validation_data),
            BestPrecisionCallback(),
        ],
    )

    return history


def _get_num_steps(
    data_generator: Optional[EdgeDG],
    steps_in_epoch: Optional[int],
    max_num_images: Optional[int],
    debug: bool,
) -> Optional[int]:
    if data_generator is None:
        return None

    if debug is True:
        num_steps = min(3, len(data_generator))
    else:
        num_steps = steps_in_epoch

        if max_num_images is not None:
            num_steps = _get_max_num_steps(max_num_images, data_generator)

    return num_steps


def _get_max_num_steps(max_num_images: int, data_generator: EdgeDG) -> int:
    num_steps = math.ceil(max_num_images / data_generator.images_in_batch)

    # check that the number of steps don't exceed length of the DataGenerator
    return min(num_steps, len(data_generator))


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

    id_in_batch = 0
    step_num = 0
    counter = 0

    num_imgs = val_data.num_data
    max_pred = min(max_pred, num_imgs * val_data.node_pairs_image)

    while counter < max_pred:
        try:
            val_x, adj_true = val_data[step_num]
        except Exception as e:
            print(
                f"Failed to get data from EdgeDG (len = {len(val_data)},",
                f"node_pairs_img = {val_data.node_pairs_image})",
                f"at step number {step_num}.",
            )
            raise Exception(e)

        skel_imgs, _, node_pair_imgs = val_x
        adj_pred, _ = classify(model_.predict(val_x))

        for (skel, np_im, a_pred, a_true) in zip(
            skel_imgs, node_pair_imgs, adj_pred, adj_true
        ):
            pair_xy = [p for p in np.fliplr(np.argwhere(np_im))]
            rgb_img = plot_node_pairs_on_skel(np.float32(skel), [pair_xy], show=show)

            table.add_data(int(a_pred), int(a_true), wandb.Image(rgb_img))

            counter += 1
            if counter >= max_pred:
                break

        # increment step
        step_num, id_in_batch = increment_step(
            step_num, id_in_batch, val_data.batch_size
        )
        if only_adj_nodes:
            # Only show predictions for adjacent nodes
            step_num, id_in_batch = choose_step_num(
                val_data, step_num=step_num, id_in_batch=id_in_batch, pick_adj=True
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
        if self.validation_data is None:
            return

        if epoch == 0:
            return

        if epoch % self.frequency == 0:
            predict(
                self.model,
                self.validation_data,
                description=f"Prediction on epoch {epoch}/{self.total_epochs}.",
            )


class BestPrecisionCallback(ModelCheckpoint):
    def __init__(self, **kwargs):
        filepath = os.path.join(
            wandb.run.dir, "weights.ep_{epoch:02d}-valprec_{val_precision:.3f}.hdf5"
        )

        super(BestPrecisionCallback, self).__init__(
            filepath,
            monitor="val_precision",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            **kwargs,
        )

        self.k = 10
        self.api_run = wandb.Api().run(wandb.run.path)
        self.best = self._get_best_smoothed()
        self.prev = self.best

        # add key to wandb summary if not already available
        self._key = f"best_{self.monitor}"
        if self._key not in wandb.summary.keys():
            self._update_wandb_summary()

        self._filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        # note: epoch starts from 0

        # parent method automatically updates self.best
        super(BestPrecisionCallback, self).on_epoch_end(epoch, logs)

        if self.monitor_op(self.best, self.prev):
            wandb.save(self._filepath)

            self.best = self._get_best_smoothed(epoch=epoch)
            self.prev = self.best

            self._update_wandb_summary()

    def _update_wandb_summary(self):
        wandb.summary[self._key] = self.best

    def _get_best_smoothed(self, epoch: Optional[int] = None) -> float:
        history = self._get_history(epoch=epoch)
        return max(smooth(history, k=self.k))

    def _get_history(self, epoch: Optional[int] = None) -> List[float]:
        history: List[dict] = self.api_run.history(keys=[self.monitor], pandas=False)

        if history:
            last_epoch = history[-1].get("_step")
            metric_history: List[float] = [h[self.monitor] for h in history]

            # wandb callback has already been called
            if last_epoch == epoch or epoch is None:
                pass
            # wandb not updated yet
            elif last_epoch == epoch - 1:
                metric_history += [self.best]
            else:
                raise Exception(
                    f"History ({len(metric_history)} entries, final epoch {last_epoch}) does not match current epoch {epoch}."
                )

            return metric_history
        else:
            return [0.0]
