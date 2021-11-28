from tools.data import get_skeletonised_ds

# user input
img_length = 256
img_channels = 3
output_channels = 2

data_path = f'/graphics/scratch/schuelej/sar/data/{img_length}'
save_path = f'/graphics/scratch/schuelej/sar/data/{img_length}/results'
use_small_dataset = True

validation_fraction = 0.1
batch_size = 3


# generated
class Config(object):
    def __init__(self):
        self.save_path = save_path
        self.batch_size = batch_size
        self.img_dims = (img_length, img_length)

        # input
        self.img_channels = 1

        # outputs
        self.output_channels = 3

        # dataset
        self.dataset = get_skeletonised_ds(data_path,
                                           shuffle=True,
                                           seed=13).take(100)

        # training vs validation
        self.num_labels = len(self.dataset)
        self.num_validation = int(validation_fraction * self.num_labels)
        self.num_train = self.num_labels - self.num_validation

        self.train_ds = iter(self.dataset.take(self.num_train))
        self.validation_ds = iter(self.dataset.skip(self.num_train))

        self.train_ids = range(self.num_train)
        self.validation_ids = range(self.num_train, self.num_labels)

        print(f'Total: {self.num_labels} training data --',
              f'[{self.num_train} training]',
              f'[{self.num_validation} validation]')
