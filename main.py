import datetime
import os

import config
from tools.data import save_results
from tools.generator import DataGenerator, TestType
from tools.plots import plot_sample_from_train_generator, plot_validation_results

base_path = "/graphics/scratch/schuelej/sar/tfgraph/"
weights_path = os.path.join(base_path, "unet_weight_model.hdf5")
log_dir = os.path.join(
    base_path, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

if __name__ == "__main__":
    conf = config.Config()

    # generate data
    training_generator = DataGenerator(conf, TestType.TRAINING, to_fit=True)
    validation_generator = DataGenerator(conf, TestType.VALIDATION, to_fit=False)

    plot_sample_from_train_generator(training_generator)

    # build model
    # pretrained_weights = weights_path if os.path.isfile(weights_path) else None
    pretrained_weights = None

    from model.unet import UNet

    unet = UNet(
        input_size=(*conf.img_dims, conf.input_channels),
        n_filters=64,
        pretrained_weights=pretrained_weights,
    )
    unet.build()

    # creating a callback, hence best weights configurations will be saved
    # model_checkpoint = unet.checkpoint(os.path.join(log_dir, 'checkpoint'))
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
    results = unet.predict(validation_generator, verbose=1)
    save_results(conf.save_path, results)
    plot_validation_results(validation_generator, results)
