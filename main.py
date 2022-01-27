import datetime
import os

import wandb

from model.unet import NodesNNExtended
from tools import Config, NodeExtractionDG, TestType, WandbConfig
from tools.plots import plot_training_sample, show_predictions

base_path = "/graphics/scratch/schuelej/sar/tfgraph/"
name = "unet_16f_extended"

time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
label = f"{name}-{time_tag}"

log_dir = os.path.join(base_path, "logs", label)
wandb_fp = os.path.join(base_path, f"configs/{name}.yaml")
weights_fp = os.path.join(base_path, f"weights_{label}.hdf5")
predict_fp = os.path.join(base_path, f"img/predict_{label}.png")

if __name__ == "__main__":
    conf = Config("config.yaml")
    network = conf.network.node_extraction

    wandb_config = WandbConfig(wandb_fp, name)
    wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=wandb_config.run_name,
        config=wandb_config.run_config,
    )

    # generate data
    training_generator = NodeExtractionDG(conf, network, TestType.TRAINING)
    validation_generator = NodeExtractionDG(conf, network, TestType.VALIDATION)
    plot_training_sample(training_generator, network=2)

    # build model
    pretrained_weights = weights_fp if os.path.isfile(weights_fp) else None
    unet = NodesNNExtended(
        input_size=(*conf.img_dims, network.input_channels),
        n_filters=wandb.config.n_filters,
        depth=wandb.config.depth,
        pretrained_weights=pretrained_weights,
    )
    unet.build()

    # callbacks
    model_checkpoint = unet.checkpoint(
        os.path.join(log_dir, "checkpoint_{epoch}.hdf5"),
        save_frequency=len(training_generator),
    )
    tensorboard_callback = unet.tensorboard_callback(log_dir)
    wandb_callback = unet.wandb_callback()

    # model training
    unet.fit(
        x=training_generator,
        steps_per_epoch=len(training_generator),
        epochs=wandb.config.epochs,
        validation_data=validation_generator,
        callbacks=[tensorboard_callback, wandb_callback, model_checkpoint],
    )
    wandb.finish()

    # saving model weights
    unet.save_weights(weights_fp)

    # display results
    show_predictions(unet, validation_generator, filepath=predict_fp)
