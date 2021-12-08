import datetime
import os

from tools import Config, DataGenerator, TestType
from tools.plots import plot_training_sample, show_predictions

base_path = "/graphics/scratch/schuelej/sar/tfgraph/"
weights_path = os.path.join(base_path, "unet_weight_model.hdf5")
time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

log_dir = os.path.join(base_path, "logs", time_tag)
predict_fp = os.path.join(base_path, f"img/predict_{time_tag}.png")

if __name__ == "__main__":
    conf = Config("config.yaml")

    # generate data
    training_generator = DataGenerator(conf, TestType.TRAINING)
    validation_generator = DataGenerator(conf, TestType.VALIDATION)

    plot_training_sample(training_generator)

    # build model
    pretrained_weights = weights_path if os.path.isfile(weights_path) else None

    from model.unet import UNet

    unet = UNet(
        input_size=(*conf.img_dims, conf.input_channels),
        n_filters=64,
        pretrained_weights=pretrained_weights,
    )
    unet.build()

    # creating a callback, hence best weights configurations will be saved
    model_checkpoint = unet.checkpoint(os.path.join(log_dir, "checkpoint"))
    tensorboard_callback = unet.tensorboard_callback(log_dir)

    # model training
    unet.fit(
        x=training_generator,
        steps_per_epoch=len(training_generator),
        epochs=5,
        callbacks=[tensorboard_callback],
    )

    # saving model weights
    unet.save_weights(weights_path)

    # display results
    show_predictions(unet, validation_generator, filepath=predict_fp)
