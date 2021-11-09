import os

from tools.files import get_sorted_imgs

# user input
img_length = 512
img_channels = 3
output_channels = 2

data_path = '/graphics/scratch/schuelej/sar/data'
dataset_name = 'GRK008'
save_path = '/graphics/scratch/schuelej/sar/data/results'
use_small_dataset = True

validation_fraction = 0.1
batch_size = 3


# generated
class Config(object):
    def __init__(self):
        self.dataset_path = os.path.join(data_path, dataset_name)
        self.save_path = save_path
        self.batch_size = batch_size
        self.img_dims = (img_length, img_length)

        # input
        self.img_channels = 3
        self.cropped_paths, self.cropped_names = get_sorted_imgs(self.dataset_path, 'cropped', use_small_dataset)

        # outputs
        self.output_channels = 2
        self.filtered_paths, self.filtered_names = get_sorted_imgs(self.dataset_path, 'filtered', use_small_dataset)
        self.skeletonised_paths, self.skeletonised_names = get_sorted_imgs(self.dataset_path, 'skeleton', use_small_dataset)

        self.num_labels = len(self.filtered_paths)
        self.num_validation = int(validation_fraction * self.num_labels)
        self.num_train = self.num_labels - self.num_validation

        self.train_ids = range(self.num_train)
        self.validation_ids = range(self.num_train, self.num_labels)

        print(f'Total: {self.num_labels} training data --',
              f'[{self.num_train} training]',
              f'[{self.num_validation} validation]')
