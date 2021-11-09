import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from PIL import Image

from tools.graph import create_graph_vec_fixed_dim, create_input_image_node_tensor
from enum import Enum, unique


@unique
class TestType(Enum):
    TRAINING = 1
    VALIDATION = 2


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, config, test_type: TestType,
                 to_fit=True, shuffle=True):
        """Initialization
        :param data_ids: list of all 'label' data_ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param img_dims: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label test_ids after every epoch
        """
        self.data_ids = config.train_ids if test_type is TestType.TRAINING else \
            config.validation_ids
        self.test_ids = np.arange(len(self.data_ids))

        # input
        self.image_names = config.cropped_names
        self.image_paths = config.cropped_paths

        # output
        self.filtered_names = config.filtered_names
        self.filtered_paths = config.filtered_paths
        self.skeletonised_names = config.skeletonised_names
        self.skeletonised_paths = config.skeletonised_paths

        # dimensions
        self.batch_size = config.batch_size
        self.img_dims = config.img_dims
        self.n_channels = config.img_channels
        self.n_classes = config.output_channels

        self.to_fit = to_fit
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.data_ids) / self.batch_size))

    def __getitem__(self, i):
        """
        Returns the i-th batch
        :param i: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # assert i < self.__len__()

        batch_ids = self.test_ids[i * self.batch_size:(i + 1) * self.batch_size]
        data_ids_for_batch = [self.data_ids[k] for k in batch_ids]

        cropped_imgs = self._generate_x_tensor(data_ids_for_batch)
        output_imgs = self._generate_y_tensor(data_ids_for_batch)

        if self.to_fit:
            return cropped_imgs, output_imgs
        else:
            return cropped_imgs

    def on_epoch_end(self):
        """ Shuffle test_ids after each epoch (default behaviour).  """
        if self.shuffle:
            np.random.shuffle(self.test_ids)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label data_ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.img_dims, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, :, :, 0] = self._load_grayscale_image(self.image_path + '/' + self.image_names[ID])
        return X

    def _generate_x_tensor(self, data_ids):
        """Generates data containing batch_size images
        :param data_ids: list of label data_ids to load
        :return: batch of images
        """
        x = np.empty((self.batch_size, *self.img_dims, self.n_channels), dtype=np.uint8)

        for i, data_id in enumerate(data_ids):
            img_path = self.image_paths[data_id]
            img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            x[i, :, :, :] = np.uint8(img_rgb)

        return np.float32(x) / 255

    def _generate_y_tensor(self, data_ids):
        """Generates data containing batch_size images
        :param data_ids: list of label data_ids to load
        :return: batch of images
        """
        y_filt = np.empty((self.batch_size, *self.img_dims, 1), dtype=np.uint8)
        y_skel = np.empty((self.batch_size, *self.img_dims, 1), dtype=np.uint8)

        for i, data_id in enumerate(data_ids):
            filtered_fp = self.filtered_paths[data_id]
            skeleton_fp = self.skeletonised_paths[data_id]

            y_filt[i, :, :, 0] = np.uint8(cv2.imread(filtered_fp, cv2.IMREAD_GRAYSCALE))
            y_skel[i, :, :, 0] = np.uint8(cv2.imread(skeleton_fp, cv2.IMREAD_GRAYSCALE))

        return [np.float32(y_filt) / 255, np.float32(y_skel) / 255]

    def _generate_pos(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label data_ids to load
        :return: batch if masks
        """
        pos = np.empty((self.batch_size, self.max_node_dim, 2))
        # adj = np.empty((self.batch_size, *self.max_adj_vec_size), dtype=int)
        image_size = cv2.imread(self.image_path + '/' + self.image_names[0]).shape

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            pos_tmp, _ = self._load_numpy_labels(self.mask_path + '/' + self.filtered_names[ID], image_size)
            pos[i, :np.size(pos_tmp[:, 0]), :] = pos_tmp
        return pos

    def _generate_adj(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label data_ids to load
        :return: batch if masks
        """
        # pos = np.empty((self.batch_size, 2, *self.max_node_dim))
        adj_vec = np.empty((self.batch_size, self.max_adj_vec_size), dtype=int)
        image_size = cv2.imread(self.image_path + '/' + self.image_names[0]).shape
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            _, adj_tmp = self._load_numpy_labels(self.mask_path + '/' + self.filtered_names[ID], image_size)
            adj_vec[i,] = create_graph_vec_fixed_dim(adj_tmp, dim_nr_nodes=self.max_node_dim)
        return adj_vec

    def _load_grayscale_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        # img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(Image.open(image_path).convert('L').resize(self.img_dims))
        return img

    def _load_numpy_labels(self, mask_path, image_size):
        # print('numpy mask path', mask_path)
        graph_label = np.load(mask_path)
        positions = graph_label[:, 0:2, 0]
        idcs_sorted_pos_1 = np.lexsort(np.fliplr(positions).T)
        # idcs_sorted_pos_1 = np.argsort(positions_tmp, axis=0)
        positions = positions[idcs_sorted_pos_1]
        positions[:, 0] = np.round((positions[:, 0] / image_size[1]) * self.img_dims[0], 0)
        positions[:, 1] = np.round((positions[:, 1] / image_size[0]) * self.img_dims[1], 0)
        idcs_sorted_pos_2 = np.lexsort(np.fliplr(
            positions).T)  # 2nd sorting just after the resizing, since some nodes could be assigned at another row due to the fact of rounding
        # positions = positions[idcs_sorted_pos_2]

        idcs_sorted_pos_fin = idcs_sorted_pos_1[idcs_sorted_pos_2]
        positions = positions[idcs_sorted_pos_1]
        adjacency = graph_label[:, 2:, 0]
        adjacency = self._permuatate4(adjacency, idcs_sorted_pos_1)
        adjacency = self._permuatate4(adjacency, idcs_sorted_pos_2)
        # adj_perm = self._permuatate4(adjacency, idcs_sorted_pos_fin)
        # adjacency = adj_perm
        return positions, adjacency

    def _permuatate(self, adj, idcs_sort):

        Id = np.identity(len(adj[:, 0]))
        idcs_sort = idcs_sort[:, 0]
        Perm = np.take(Id, idcs_sort, axis=0)
        adj_perm = Perm @ adj @ np.transpose(Perm)
        return adj_perm

    def _permuatate2(self, adj, idcs_sort):
        pVec0 = idcs_sort[:, 0]
        # pVec1 =  idcs_sort[:,1]
        adj = adj.take(pVec0, axis=0, out=adj)
        adj = adj.take(pVec0, axis=1, out=adj)
        return adj

    def _permuatate3(self, adj, idcs_sort):
        pVec0 = idcs_sort[:, 0]
        pVec1 = idcs_sort[:, 1]
        adj[:, :] = adj[pVec0, :]
        adj[:, :] = adj[:, pVec0]
        return adj

    def _permuatate4(self, adj, idcs_sort):
        Id = np.identity(len(adj[:, 0]))
        idcs_sort = idcs_sort
        Perm = np.take(Id, idcs_sort, axis=0)
        adj_perm = Perm @ adj @ np.transpose(Perm)
        return adj_perm
