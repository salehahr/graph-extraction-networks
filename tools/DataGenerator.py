from __future__ import annotations

import os.path
import random
from abc import ABC
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from tools import EdgeDGSingle
from tools import data as data_op
from tools.adj_matr import transform_adj_matr
from tools.files import shorten_filepath
from tools.TestType import TestType

if TYPE_CHECKING:
    from tools.config import Config, RunConfig


GraphBatchData = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
GraphBatchDatasets = Tuple[tf.data.Dataset, ...]


def get_nedg(
    config: Config,
    augmented: bool = True,
    shuffle: bool = True,
):
    return {
        test: NodeExtractionDG(
            config,
            config.network.node_extraction,
            test,
            augmented=augmented,
            shuffle=shuffle,
        )
        for test in TestType
    }


def get_gedg(
    config: Config,
    batch_size: Optional[int] = None,
    augmented: bool = True,
    shuffle: bool = True,
) -> Dict[TestType, GraphExtractionDG]:
    g_network = config.network.graph_extraction
    orig_batch_size = config.batch_size

    if batch_size is not None:
        config.batch_size = batch_size

    graph_data = {
        test: GraphExtractionDG(
            config, g_network, test, augmented=augmented, shuffle=shuffle
        )
        for test in TestType
    }

    if batch_size is not None:
        config.batch_size = orig_batch_size

    return graph_data


def get_eedg_multiple(
    config: Config,
    run_config: RunConfig,
    graph_data: Optional[Dict[TestType, GraphExtractionDG]] = None,
    with_path: Optional[bool] = False,
) -> Dict[TestType, EdgeDGMultiple]:
    """Returns training/validation data for Edge NN."""
    graph_data = get_gedg(config) if graph_data is None else graph_data

    return {
        test: EdgeDGMultiple(config, run_config, graph_data[test], with_path=with_path)
        for test in TestType
    }


def get_eedg(
    config: Config,
    run_config: RunConfig,
    with_path: Optional[bool] = False,
    validate: bool = True,
    debug: bool = False,
    test: bool = False,
) -> Union[EdgeDG, Dict[TestType, EdgeDG]]:
    """Returns training/validation data for Edge NN."""
    # don't shuffle dataset debug mode
    shuffle = not debug
    if test:
        validate = False
        shuffle = False

    if validate:
        return {
            test: EdgeDG(
                config,
                run_config,
                with_path=with_path,
                test_type=test,
                shuffle=shuffle,
                debug=debug,
            )
            for test in TestType
        }
    elif test:
        return EdgeDG(
            config,
            run_config,
            with_path=with_path,
            test_type=TestType.TESTING,
            shuffle=shuffle,
        )
    else:
        return {
            TestType.TRAINING: EdgeDG(
                config,
                run_config,
                with_path=with_path,
                test_type=TestType.TRAINING,
                shuffle=shuffle,
            )
        }


class DataGenerator(tf.keras.utils.Sequence, ABC):
    """Generates batches of training/validation/test data."""

    def __init__(
        self,
        config,
        network,
        test_type: TestType,
        augmented: bool = True,
        shuffle: bool = True,
        debug: bool = False,
    ):
        # dataset settings
        self.test_type = test_type
        self.shuffle = shuffle
        self.config = config

        if test_type == TestType.TRAINING:
            num_data = config.num_train
            if debug:
                ds = data_op.get_debug_ds()
            else:
                ds = config.training_ds

        elif test_type == TestType.VALIDATION:
            num_data = config.num_validation
            ds = config.validation_ds
        elif test_type == TestType.TESTING:
            num_data = config.num_test
            ds = config.test_ds
        else:
            raise Exception

        # shuffle dataset before batching
        if shuffle:
            ds = ds.shuffle(num_data, reshuffle_each_iteration=False)
        self.ds = ds.batch(config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)

        # dimensions
        self.num_data = num_data
        self.batch_size = config.batch_size
        self.img_dims = config.img_dims

        # network attributes
        self.network = network.id
        self.input_channels = network.input_channels
        self.output_channels = network.output_channels

        # augment data when not performing validation
        if test_type == TestType.VALIDATION:
            self.augmented = False
        else:
            self.augmented = augmented

        # shuffle
        self.shuffle_buffer = self.num_data
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        i.e. number of steps per epoch."""
        return int(np.floor(self.num_data / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.ds = self.ds.shuffle(
                self.shuffle_buffer, reshuffle_each_iteration=False
            )

    def _get_data(
        self, i: Optional[int] = None, return_fps: bool = False
    ) -> GraphBatchDatasets:
        skel_fps, graph_fps = self._get_fps(i)

        skel_imgs = skel_fps.map(
            data_op.fp_to_grayscale_img, num_parallel_calls=tf.data.AUTOTUNE
        )
        node_pos, degrees, node_types, adj_matr = self._get_graph_data(graph_fps)

        def set_shape(img: tf.Tensor) -> tf.Tensor:
            img.set_shape([*self.img_dims, 1])
            return img

        skel_imgs = skel_imgs.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)
        node_pos = node_pos.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)
        degrees = degrees.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)
        node_types = node_types.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)

        if return_fps:
            return skel_imgs, node_pos, degrees, node_types, adj_matr, skel_fps
        else:
            return skel_imgs, node_pos, degrees, node_types, adj_matr

    def _get_fps(self, i: Optional[int] = None):
        """
        Returns filepaths of the skeletonised images and graphs.
        :param i: batch id (optional)
        """

        skel_fps = self.ds if i is None else self.ds.skip(i).take(1)
        skel_fps = skel_fps.unbatch()

        def skel_to_graph(skel_fp: tf.string):
            graph_fp = tf.strings.regex_replace(skel_fp, "skeleton", "graphs")
            graph_fp = tf.strings.regex_replace(graph_fp, r"\.png", ".json")
            return graph_fp

        graph_fps = skel_fps.map(skel_to_graph, num_parallel_calls=tf.data.AUTOTUNE)

        return skel_fps, graph_fps

    def _get_graph_data(
        self, graph_fps: tf.data.Dataset
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Generates node attribute tensors from graph.
        :param graph_fps: filepaths to the graph objects
        :return: node attributes and adjacency vector
        """

        node_attrs = graph_fps.map(self._to_node_attributes).unbatch()
        adj_matr = graph_fps.map(data_op.tf_fp_to_adj_matr).map(
            lambda x: tf.cast(x, tf.int32)
        )

        node_pos = node_attrs.window(1, shift=3).flat_map(lambda x: x)
        degrees = node_attrs.skip(1).window(1, shift=3).flat_map(lambda x: x)
        node_types = node_attrs.skip(2).window(1, shift=3).flat_map(lambda x: x)

        return node_pos, degrees, node_types, adj_matr

    def _to_node_attributes(self, fp):
        return tf.numpy_function(
            func=data_op.fp_to_node_attributes,
            inp=[fp, self.img_dims[0]],
            Tout=tf.uint8,
        )

    def _augment(self, data: List[tf.data.Dataset]):
        seed1 = random.randint(0, 100)
        seed2 = random.randint(0, 100)

        def augment(x):
            return self._flip_tensor(x, seeds=(seed1, seed2))

        return [d.map(augment, deterministic=True) for d in data]

    @staticmethod
    def _flip_tensor(x: tf.Tensor, seeds: tuple) -> tf.Tensor:
        x = tf.image.random_flip_left_right(x, seed=seeds[0])
        x = tf.image.random_flip_up_down(x, seed=seeds[1])
        return x

    def _rebatch(self, data: List[tf.data.Dataset]) -> List[tf.Tensor]:
        return [data_op.rebatch(d, self.batch_size) for d in data]


class NodeExtractionDG(DataGenerator):
    def __getitem__(
        self, i: int
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        """
        Returns the i-th batch.
        :param i: batch index
        :return: skeletonised images and node attributes of the images in the batch
        """
        skel_imgs, node_pos, degrees, node_types, _ = self._get_data(i)

        # augment
        data = [skel_imgs, node_pos, degrees, node_types]
        data = self._augment(data) if self.augmented else data

        # rebatch
        skel_imgs, node_pos, degrees, node_types = self._rebatch(data)

        return skel_imgs, (node_pos, degrees, node_types)


class GraphExtractionDG(DataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_nodes: int = 300
        self.max_adj_dim: int = int(self.max_nodes * (self.max_nodes - 1) / 2)

    def __getitem__(
        self, i: int, return_filepaths: bool = False
    ) -> Union[
        Tuple[GraphBatchData, tf.Tensor, List[str]],
        Tuple[GraphBatchData, tf.Tensor],
    ]:
        """Returns the i-th batch."""
        data_ = self._get_data(i, return_fps=return_filepaths)
        (skel_imgs, node_pos, degrees), adj_matrs = data_[0:3], data_[4]

        pos_idx_img = self._to_pos_indices_img(node_pos)

        # augment
        data = [skel_imgs, node_pos, degrees, pos_idx_img]
        data = self._augment(data) if self.augmented else data

        pos_idx_aug_img = data[-1]
        adj_matrs = self._augment_adj_matr(adj_matrs, pos_idx_aug_img)

        # rebatch
        data = [*data[:3], adj_matrs]
        skel_imgs, node_pos, degrees, adj_matrs = self._rebatch(data)

        if return_filepaths:
            filepaths = data_op.ds_to_list(data_[-1])
            return (skel_imgs, node_pos, degrees), adj_matrs, filepaths
        else:
            return (skel_imgs, node_pos, degrees), adj_matrs

    def get_single_data_point(self, i: int) -> Tuple[Tuple, tf.RaggedTensor, str]:
        """Returns data for a batch size of one,
        with squeezed dimensions for convenience."""
        assert self.batch_size == 1

        (skel_img, node_pos, degrees), adj_matr_true, filepaths = self.__getitem__(
            i, return_filepaths=True
        )

        # squeeze [1, 256, 256, 1] to [256, 256, 1]
        skel_img = tf.squeeze(skel_img, axis=0)
        node_pos = tf.squeeze(node_pos, axis=0)
        degrees = tf.squeeze(degrees, axis=0)

        filepath = shorten_filepath(filepaths[0], data_path=self.config.data_path)

        return (skel_img, node_pos, degrees), adj_matr_true, filepath

    def _to_pos_indices_img(self, node_pos: tf.data.Dataset) -> tf.data.Dataset:
        """Generates an image dataset containing the integer indices
        of the node positions, at the corresponding (x,y) coordinates."""

        # get pos list from node_pos image
        pos_list = node_pos.map(
            data_op.sorted_pos_list_from_image, num_parallel_calls=tf.data.AUTOTUNE
        )

        idx = pos_list.map(
            data_op.indices_for_pos_list, num_parallel_calls=tf.data.AUTOTUNE
        )

        return tf.data.Dataset.zip((idx, pos_list)).map(
            lambda i, xy: data_op.tf_pos_indices_image(i, xy, self.img_dims[0]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    @staticmethod
    def _augment_adj_matr(
        adj_matr: tf.data.Dataset, pos_idx_aug_img: tf.data.Dataset
    ) -> tf.data.Dataset:
        """Transforms adjacency matrix based on new node positions
        and returns it in RaggedTensor format."""

        def get_idx(img):
            return tf.numpy_function(
                func=data_op.get_data_at_xy, inp=[img], Tout=tf.uint32
            )

        aug_sort_indices = pos_idx_aug_img.map(get_idx)

        A_pos = tf.data.Dataset.zip((adj_matr, aug_sort_indices))
        return A_pos.map(transform_adj_matr, num_parallel_calls=tf.data.AUTOTUNE)

    @property
    def all_data(self):
        return self._get_data()


class EdgeDG(GraphExtractionDG):
    def __init__(
        self,
        config: Config,
        run_config: RunConfig,
        with_path: bool,
        test_type: TestType,
        **kwargs,
    ):
        super(EdgeDG, self).__init__(
            config, config.network.graph_extraction, test_type, **kwargs
        )

        self.config = config
        self.with_path = with_path

        # network attributes
        network = config.network.edge_extraction
        self.network = network.id
        self.input_channels = network.input_channels
        self.output_channels = network.output_channels

        # update batch size
        self.images_in_batch: int = run_config.images_in_batch
        self.node_pairs_image: int = run_config.node_pairs_in_image
        self.node_pairs_batch: int = self.images_in_batch * self.node_pairs_image
        self.batch_size: int = self.node_pairs_batch

        # shuffle
        self.shuffle_buffer = min(self.num_data, run_config.parameters.train_imgs)
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        i.e. number of steps per epoch."""
        return int(np.floor(self.num_data / self.images_in_batch))

    def __getitem__(self, item: int, **kwargs) -> Tuple:
        (
            (skel_imgs, node_positions, degrees),
            adj_matrs,
            filepaths,
        ) = super().__getitem__(item, return_filepaths=True)

        # placeholders for data across images in gedg batch
        # not using self.images_in_batch here in case batch gets truncated
        num_images: int = skel_imgs.shape[0]
        node_pair_imgs: List[Optional] = [None] * num_images
        num_nodes_per_image: List[Optional] = [None] * num_images
        adjacencies = [None] * num_images
        paths = [None] * num_images

        # iterate over the images (self.images_in_batch)
        for i, data in enumerate(zip(skel_imgs, node_positions, adj_matrs)):
            skel_img, node_pos, adj_matr = data

            # derived data
            pos_list = data_op.sorted_pos_list_from_image(node_pos)

            # there should be enough combos to create a batch
            num_nodes = tf.shape(pos_list)[0]
            combos = self._get_combos(num_nodes, adj_matr)
            self._assert_enough_combos(combos, item, i, filepaths[i], num_images)

            # iterate over node combinations (self.node_pairs_in_image)
            batch_combo = combos[0 : self.node_pairs_image]
            x, y = self._get_combo_imgs_and_labels(
                skel_img, batch_combo, adj_matr, pos_list
            )

            node_pair_imgs[i] = x
            num_nodes_per_image[i] = tf.shape(x)[0]

            if self.with_path:
                adjacencies[i], paths[i] = y
            else:
                adjacencies[i] = y

        # actual outputs of the data generator
        """
        images_in_batch x node_pairs_in_image combinations.
        e.g. for
            images_in_batch = 3
            node_pairs_in_image = 4
            ----------
            effective_batch_size = 12

        size(combo_imgs) = 12 x 256 x 256 x 2
        size(adjacencies) = 12 x 1
        """
        node_pair_imgs = tf.concat(node_pair_imgs, axis=0)
        adjacencies = tf.concat(adjacencies, axis=0)

        # repeat skel_imgs and node_positions values to match dimensions of node_pair_imgs
        skel_imgs_in_batch = data_op.repeat_to_match_dims(
            skel_imgs, num_nodes_per_image
        )
        node_positions_in_batch = data_op.repeat_to_match_dims(
            node_positions, num_nodes_per_image
        )

        if self.with_path:
            paths = tf.concat(paths, axis=0)
            return (skel_imgs_in_batch, node_positions_in_batch, node_pair_imgs), (
                adjacencies,
                paths,
            )
        else:
            return (
                skel_imgs_in_batch,
                node_positions_in_batch,
                node_pair_imgs,
            ), adjacencies

    def _get_combos(self, num_nodes: tf.Tensor, adj_matr: tf.Tensor) -> tf.Tensor:
        # shuffle before batching -- todo: set seed for np.random
        combos = data_op.get_all_node_combinations(tf.cast(num_nodes, tf.int64))
        combos = data_op.get_reduced_node_combinations(
            combos,
            adj_matr,
            shuffle=self.shuffle,
            test_type=self.test_type,
            adjacency_fraction=self.config.adjacency_fraction,
        )
        return combos

    def _assert_enough_combos(
        self,
        combos: tf.Tensor,
        batch_num: int,
        i: int,
        current_filepath: str,
        num_images: int,
    ) -> None:
        try:
            assert len(combos) > self.node_pairs_image
        except AssertionError as e:
            print(
                f"Not enough node combos found! Need {self.node_pairs_image} combos,"
                f"but only {len(combos)} were extracted.\n"
            )
            print(
                f"This was at batch number {batch_num} (0-idx),"
                f"out of {len(self)} total steps.\n",
            )
            print(
                f"\t At {i}-th (0-idx) image in batch,\n" f"\t\t {current_filepath}\n",
                f"\t out of {num_images} images in batch.",
            )
            raise AssertionError(e)

    def _get_combo_imgs_and_labels(
        self,
        skel_img: tf.Tensor,
        batch_combo: tf.Tensor,
        adj_matr: tf.Tensor,
        pos_list: tf.Tensor,
    ) -> Tuple[
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
    ]:
        # input data
        combo_imgs = data_op.get_combo_imgs(batch_combo, skel_img, pos_list)

        # labels
        adj = tf.stack([data_op.get_combo_adjacency(c, adj_matr) for c in batch_combo])
        adj = tf.reshape(adj, [tf.shape(adj)[0], 1])

        if self.with_path:
            path = [
                data_op.get_combo_path(c, a, pos_list, skel_img)
                for (a, c) in zip(adj, batch_combo)
            ]
            path = tf.stack(path)
            y = (adj, path)
        else:
            y = adj

        return combo_imgs, y


class EdgeDGMultiple(tf.keras.utils.Sequence):
    """
    Generates node combinations and corresponding labels (path and adjacency)
    across multiple skeletonised images.

    One batch contains several images.
    - One image has n = node_pairs_in_image combinations.
    - One image has one pos_list.

    e.g. for
        total_images = 24
        images_in_batch = 3
        node_pairs_in_image = 4
        ----------
        num_batches = 24 / 3 = 8
        effective_batch_size = 3 * 4 = 12

    self.combos[batch_num] = [
        0: [12x2] as tf.Tensor
        1: [12x2]
        .
        .
        8: [12x2]
    ]

    self.pos_list[batch_num][img_num_in_batch] = [
        0:  [
                0: [pos...]     of img 0 in batch, given as tf.Tensor
                1: [pos...]
                2: [pos...]
            ]
        .
        .
        8: [ ... ]
    ]
    """

    def __init__(
        self,
        config: Config,
        run_config: RunConfig,
        gedg: GraphExtractionDG,
        with_path: bool,
    ):
        self.config = config

        self.test_type = gedg.test_type
        self.with_path = with_path

        self.gedg = gedg
        self.num_batches = len(self.gedg)

        # dimensions
        network = config.network.edge_extraction
        self.input_channels = network.input_channels
        self.output_channels = network.output_channels
        self.img_dims = config.img_dims

        # batch size
        self.images_in_batch: int = run_config.images_in_batch
        self.node_pairs_image: int = run_config.node_pairs_in_image
        self.node_pairs_batch: int = self.images_in_batch * self.node_pairs_image

        # shuffle
        self.on_epoch_end()

        # checks
        assert self.gedg.batch_size == self.batch_size

    @property
    def batch_size(self) -> int:
        return self.images_in_batch

    @property
    def effective_batch_size(self) -> int:
        return self.node_pairs_batch

    def on_epoch_end(self):
        self.gedg.on_epoch_end()

    def __len__(self):
        """One step in an epoch corresponds to the same step in the GEDG object."""
        return self.num_batches

    def __getitem__(
        self, i: int
    ) -> Tuple[tf.Tensor, Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]]:
        (skel_imgs, node_positions, degrees), adj_matrs = self.gedg[i]

        # placeholders for data across images in gedg batch
        num_images = skel_imgs.shape[0]
        combo_imgs = [None] * num_images
        adjacencies = [None] * num_images
        paths = [None] * num_images

        # iterate over the images (num images = self.batch_size)
        for ii, data in enumerate(zip(skel_imgs, node_positions, degrees, adj_matrs)):
            skel_img, node_pos, degree, adj_matr = data

            # eedg_single returns node combinations (n = node_pairs_image)
            eedg_single = EdgeDGSingle(
                self.config,
                self.node_pairs_image,
                self.test_type,
                skel_img,
                node_pos,
                degree,
                adj_matr,
                with_path=self.with_path,
                seed=i + ii,
            )

            x, y = eedg_single[0]

            combo_imgs[ii] = x
            if self.with_path:
                adjacencies[ii], paths[ii] = y
            else:
                adjacencies[ii] = y

        # actual outputs of the data generator
        """
        images_in_batch x node_pairs_in_image combinations.
        e.g. for
            images_in_batch = 3
            node_pairs_in_image = 4
            ----------
            effective_batch_size = 12

        size(combo_imgs) = 12 x 256 x 256 x 2
        size(adjacencies) = 12 x 1
        """
        combo_imgs = tf.concat(combo_imgs, axis=0)
        adjacencies = tf.concat(adjacencies, axis=0)

        if self.with_path:
            paths = tf.concat(paths, axis=0)
            return combo_imgs, (adjacencies, paths)
        else:
            return combo_imgs, adjacencies
