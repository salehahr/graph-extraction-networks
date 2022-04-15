import os
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

from tools.PolyGraph import PolyGraph
from tools.sort import get_sort_indices, sort_list_of_nodes
from tools.TestType import TestType


def get_debug_ds():
    files_glob = [
        os.path.join("/graphics/scratch/schuelej/sar/data/debug/*/skeleton/*.png"),
        os.path.join("/graphics/scratch/schuelej/sar/data/debug/*/*/skeleton/*.png"),
    ]
    return tf.data.Dataset.list_files(files_glob, shuffle=False)


def get_skeletonised_ds(
    data_path: str, seed: int, is_test: bool = False
) -> tf.data.Dataset:
    """
    Returns skeletonised image paths.
    :param data_path: folder where the data is stored
    :param seed: seed for shuffling function, set None for random
    :param is_test: True if test set filepaths are to be used
    :return: the skeletonised image paths as a Dataset object
    """
    if is_test:
        skeletonised_files_glob = [
            os.path.join(data_path, "test_*/skeleton/*.png"),
            os.path.join(data_path, "*/test_*/skeleton/*.png"),
        ]
    else:
        skeletonised_files_glob = [
            os.path.join(data_path, "[!t]*/skeleton/*.png"),
            os.path.join(data_path, "*/[!t]*/skeleton/*.png"),
        ]

    ds = tf.data.Dataset.list_files(skeletonised_files_glob, shuffle=False)

    return ds.shuffle(len(ds), seed=seed, reshuffle_each_iteration=False)


def filter_ds_synth(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.filter(lambda x: tf.strings.regex_full_match(x, ".*GRK.*"))


def ds_to_list(dataset: tf.data.Dataset) -> List[str]:
    return [f.decode("utf-8") for f in dataset.as_numpy_iterator()]


@tf.function
def fp_to_grayscale_img(fp: tf.Tensor) -> tf.Tensor:
    raw_img = tf.io.read_file(fp)
    unscaled_img = tf.image.decode_png(raw_img, channels=1, dtype=tf.uint8)
    return tf.image.convert_image_dtype(unscaled_img, tf.float32)


def fp_to_node_attributes(fp: str, dim: int) -> np.ndarray:
    graph = PolyGraph.load(fp)
    return graph_to_node_attributes(graph, dim)


def graph_to_node_attributes(graph: PolyGraph, dim: int) -> np.ndarray:
    """
    Generates output matrices of the graph's node attributes.
    """
    node_attributes = np.zeros((3, dim, dim, 1)).astype(np.uint8)

    node_pos = node_attributes[0, :, :, :]
    degrees = node_attributes[1, :, :, :]
    node_types = node_attributes[2, :, :, :]

    for i, (col, row) in enumerate(graph.positions):
        node_pos[row][col] = 1
        degrees[row][col] = graph.num_node_neighbours[i]
        node_types[row][col] = graph.node_types[i]

    def cap_degrees(deg_matrix: np.ndarray) -> np.ndarray:
        """Cap values at 4."""
        cap_value = 4
        deg_matrix[deg_matrix > cap_value] = cap_value
        return deg_matrix

    degrees[:, :, :] = cap_degrees(degrees)

    return node_attributes


def fp_to_adj_matr(fp: str) -> np.ndarray:
    adj_matr = PolyGraph.load(fp).adj_matrix.astype(np.uint8)
    adj_matr = np.expand_dims(adj_matr, -1)
    return adj_matr


def tf_fp_to_adj_matr(fp: str):
    return tf.numpy_function(func=fp_to_adj_matr, inp=[fp], Tout=tf.uint8)


def pos_list_from_image(node_pos_img: np.ndarray) -> np.ndarray:
    """Gets list of coordinates from the node_pos image."""
    # flip to convert (row, col) to (x, y)
    pos_list_xy = np.fliplr(np.argwhere(node_pos_img)).tolist()
    return sort_list_of_nodes(pos_list_xy)


@tf.function
def sorted_pos_list_from_image(node_pos_img: tf.Tensor) -> tf.Tensor:
    """Extracts the sorted xy coordinates from the node_pos image."""
    xy_unsorted = unsorted_pos_list_from_image(node_pos_img)
    sort_indices = get_sort_indices(xy_unsorted)
    return tf.gather(xy_unsorted, sort_indices)


@tf.function
def indices_for_pos_list(pos_list: tf.Tensor) -> tf.Tensor:
    """ "Returns the indices that correspond to each node in the pos_list.
    Generates a range of values [0, n_nodes) = original node indices."""
    n_nodes = tf.shape(pos_list)[0]
    return tf.range(start=0, limit=n_nodes)


def gen_pos_indices_img(idx: np.ndarray, xy: np.ndarray, dim: int) -> np.ndarray:
    """Generates an image containing the integer indices of the node positions,
    at the corresponding (x,y) coordinates."""

    # placeholder data type must be bigger than uint8
    # -- if uint8, overflow can happen if there are more than 255 nodes
    img = np.zeros((dim, dim, 1)).astype(np.uint32)

    for i, (col, row) in enumerate(xy):
        img[row, col, :] = idx[i] + 1  # add one to avoid losing first index

    return img


def tf_pos_indices_image(i, xy, img_length):
    return tf.numpy_function(
        func=gen_pos_indices_img, inp=[i, xy, img_length], Tout=tf.uint32
    )


@tf.function
def unsorted_pos_list_from_image(node_pos_img: tf.Tensor) -> tf.Tensor:
    """Extracts the (unsorted) xy coordinates from the node_pos image."""
    node_pos_img = tf.cast(tf.squeeze(node_pos_img), tf.uint8)
    return tf.reverse(tf.where(node_pos_img), axis=[1])


def get_data_at_xy(matr: np.ndarray) -> np.ndarray:
    """Extracts data from a 2D matrix at the given (x,y) coordinate."""
    matr = matr.squeeze()
    rc = np.fliplr(pos_list_from_image(matr))
    return matr[rc[:, 0], rc[:, 1]] - 1


def rc_to_node_combo_img(
    rc1: np.ndarray, rc2: np.ndarray, dims: np.ndarray
) -> np.ndarray:
    """ "Converts a pair of (row, col) coordinates to a blank image
    with white dots corresponding to the given coordinates."""
    img = np.zeros(dims, dtype=np.int64)

    img[rc1[0], rc1[1]] = 1
    img[rc2[0], rc2[1]] = 1

    return img


@tf.function
def tf_rc_to_node_combo_img(rc1, rc2, img_dims):
    return tf.numpy_function(
        rc_to_node_combo_img, inp=(rc1, rc2, img_dims), Tout=tf.int64
    )


@tf.function
def tf_xy_to_node_combo_img(xy1, xy2, img_dims):
    rc1 = tf.reverse(xy1, axis=[-1])
    rc2 = tf.reverse(xy2, axis=[-1])
    return tf.numpy_function(
        rc_to_node_combo_img, inp=(rc1, rc2, img_dims), Tout=tf.int64
    )


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int64)])
def get_all_node_combinations(num_nodes: tf.Tensor) -> tf.Tensor:
    indices = tf.range(num_nodes)

    # meshgrid to create all possible pair combinations
    a, b = tf.meshgrid(indices, indices)
    a = tf.linalg.band_part(a, 0, -1)
    b = tf.linalg.band_part(b, 0, -1)
    grid = tf.stack((a, b), axis=-1)
    combos = tf.reshape(grid, shape=(num_nodes * num_nodes, 2))

    # remove pairs where both nodes are the same
    idcs_not_equal = tf.where(tf.not_equal(combos[:, 0], combos[:, 1]))
    combos = tf.gather_nd(combos, idcs_not_equal)

    return combos


def get_reduced_node_combinations(
    all_combos: tf.Tensor,
    adj_matr: tf.Tensor,
    shuffle: bool = True,
    test_type: TestType = TestType.TRAINING,
    adjacency_fraction: Optional[float] = None,
) -> tf.Tensor:
    """Returns a dataset of node combinations that have equal amounts of
    adjacent and non-adjacent node pairs. The dataset elements are
    in alternating order: adj > not adj > adj > not adj > ..."""
    adjacencies = [(pair, get_combo_adjacency(pair, adj_matr)) for pair in all_combos]

    # categorise according to adjacency
    adj_combos = [pair for (pair, adj) in adjacencies if adj == 1]
    non_adj_combos = [pair for (pair, adj) in adjacencies if adj == 0]

    # use random combinations when testing
    if test_type == TestType.TESTING:
        shuffle = True

    if shuffle is True:
        np.random.shuffle(adj_combos)
        np.random.shuffle(non_adj_combos)

    # training ds: 1 to 1 ratio of adjacent and non-adjacent nodes
    if (
        test_type == TestType.TRAINING
        or test_type == TestType.TESTING
        or adjacency_fraction is None
        or adjacency_fraction == 0.5
    ):
        # zip + nested list comprehension = interleaving while cutting off excess non_adj_nodes
        reduced_combos = [
            node
            for adj_and_non_adj in zip(adj_combos, non_adj_combos)
            for node in adj_and_non_adj
        ]
    else:
        num_combos = len(adjacencies)
        num_adj_soll = int(adjacency_fraction * num_combos)

        # too few adjacencies
        if len(adj_combos) < num_adj_soll:
            num_adj = len(adj_combos)
            num_combos = num_adj / adjacency_fraction
        else:
            num_adj = num_adj_soll
        num_non_adj = int((1 - adjacency_fraction) * num_combos)

        reduced_combos = adj_combos[:num_adj] + non_adj_combos[:num_non_adj]
        np.random.shuffle(reduced_combos)

    return tf.stack(reduced_combos)


@tf.function
def node_pair_to_coords(
    pair: tf.Tensor, pos_list: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    xy1 = pos_list[pair[0], :]
    xy2 = pos_list[pair[1], :]

    rc1 = tf.reverse(xy1, axis=[0])
    rc2 = tf.reverse(xy2, axis=[0])

    return rc1, rc2


def get_combo_imgs(
    batch_combo: tf.Tensor,
    skel_img: tf.Tensor,
    pos_list: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    skel_img = tf.squeeze(skel_img)

    coords = [node_pair_to_coords(p, pos_list) for p in batch_combo]
    node_pair_imgs = [
        rc_to_node_combo_img(rc1, rc2, skel_img.shape) for (rc1, rc2) in coords
    ]

    return tf.stack(node_pair_imgs)


def get_combo_imgs_from_xy(
    combos_xy: tf.Tensor,
    img_dims: tf.Tensor,
) -> tf.data.Dataset:
    combos_rc = tf.reverse(combos_xy, axis=[-1])
    ds = tf.data.Dataset.from_tensor_slices(combos_rc)
    return ds.map(
        lambda x: tf.numpy_function(
            rc_to_node_combo_img, inp=(x[0], x[1], img_dims), Tout=tf.int64
        )
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(256, 256), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256), dtype=tf.uint8),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
    ],
)
def get_combo_inputs(
    skel_img: tf.Tensor,
    node_pos_img: tf.Tensor,
    combos: tf.Tensor,
    pos_list_xy: tf.Tensor,
) -> tf.data.Dataset:
    img_dims = tf.shape(skel_img)
    combos_xy = tf.gather(pos_list_xy, combos)
    combo_imgs = get_combo_imgs_from_xy(combos_xy, img_dims)
    num_neighbours = tf.cast(tf.shape(combos_xy)[0], tf.int64)

    return (
        combo_imgs.map(lambda x: (skel_img, node_pos_img, x))
        .batch(num_neighbours)
        .get_single_element()
    )


@tf.function
def get_combo_adjacency(pair: tf.Tensor, adj_matr: tf.Tensor) -> tf.Tensor:
    n1, n2 = pair[0], pair[1]
    return adj_matr[n1, n2]


@tf.function
def get_combo_path(
    pair: tf.Tensor, adjacency: tf.Tensor, pos_list: tf.Tensor, skel_img: tf.Tensor
) -> tf.Tensor:
    rc1, rc2 = node_pair_to_coords(pair, pos_list)
    row_indices = tf.sort([rc1[0], rc2[0]])
    col_indices = tf.sort([rc1[1], rc2[1]])

    img_section = skel_img[
        row_indices[0] : row_indices[1] + 1,
        col_indices[0] : col_indices[1] + 1,
    ]

    return tf.math.multiply(
        tf.cast(adjacency, tf.float32),
        tf.RaggedTensor.from_tensor(img_section),
    )


def repeat_to_match_dims(imgs: tf.Tensor, dims: List[tf.Tensor]) -> tf.Tensor:
    imgs_per_combo = [tf.stack([im for _ in range(n)]) for n, im in zip(dims, imgs)]
    imgs_in_batch = tf.concat(imgs_per_combo, axis=0)

    return imgs_in_batch


@tf.function
def rebatch(x: tf.data.Dataset, batch_size: int) -> tf.Tensor:
    """Converts tf.data.Dataset object to a tf.Tensor object"""
    return (
        x.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        .take(1)
        .get_single_element()
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.uint8),
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.uint8),
    ]
)
def data_from_node_imgs(
    node_pos: tf.Tensor, degrees: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Returns lookup table consisting of node xy position and the corresponding degree."""
    pos_list_xy = sorted_pos_list_from_image(node_pos)
    pos_list_rc = tf.reverse(pos_list_xy, axis=[1])
    degrees_list = tf.gather_nd(indices=pos_list_rc, params=tf.squeeze(degrees))
    num_nodes = tf.shape(pos_list_xy, out_type=tf.int64)[0]

    return pos_list_xy, degrees_list, num_nodes


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.int64),
    ],
)
def unique_2d(tensor: tf.Tensor) -> tf.Tensor:
    """Since tf.unique is currently for 1D tensors,
    use this for finding unique values across 2D or more."""
    return tf.numpy_function(
        lambda x: np.unique(x, axis=0), inp=[tensor], Tout=tf.int64
    )


@tf.function
def get_rc_ragged(ragged, flat_idx):
    """Converts the flat index to the [row, col] format."""
    row = tf.where(ragged.row_splits <= flat_idx)[-1, 0]

    row_split = tf.squeeze(tf.gather(ragged.row_splits, row))
    col = tf.squeeze(tf.math.floormod(flat_idx, row_split))

    return tf.stack((row, col))


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int64)])
def is_empty_tensor(tensor: tf.Tensor) -> tf.bool:
    return tf.equal(tf.size(tensor), 0)
