import os
from typing import Tuple

import numpy as np
import tensorflow as tf

from tools.PolyGraph import PolyGraph
from tools.sort import get_sort_indices, sort_list_of_nodes


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


def ds_to_list(dataset: tf.data.Dataset) -> list:
    return [f.decode("utf-8") for f in dataset.as_numpy_iterator()]


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


def pos_list_from_image(node_pos_img: np.ndarray) -> np.ndarray:
    """Gets list of coordinates from the node_pos image."""
    # flip to convert (row, col) to (x, y)
    pos_list_xy = np.fliplr(np.argwhere(node_pos_img)).tolist()
    return sort_list_of_nodes(pos_list_xy)


def sorted_pos_list_from_image(node_pos_img: tf.Tensor) -> tf.Tensor:
    """Extracts the sorted xy coordinates from the node_pos image."""
    xy_unsorted = unsorted_pos_list_from_image(node_pos_img)
    sort_indices = get_sort_indices(xy_unsorted)
    return tf.gather(xy_unsorted, sort_indices)


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


def get_all_node_combinations(num_nodes: tf.Tensor) -> tf.Tensor:
    indices = tf.range(num_nodes)

    # meshgrid to create all possible pair combinations
    y, x = tf.meshgrid(indices, indices)
    x = tf.expand_dims(x, 2)
    y = tf.expand_dims(y, 2)
    z = tf.concat([x, y], axis=2)

    # extract the upper triangular part of the meshgrid
    all_combos = tf.constant([[0, 0]])  # placeholder
    for x in range(num_nodes - 1):
        # goes from 0 to n - 2
        aa = z[x + 1 :, x, :]
        all_combos = tf.concat([all_combos, aa], axis=0)
    all_combos = all_combos[1:, :]  # remove initial [0, 0] combination

    return all_combos


def get_reduced_node_combinations(
    all_combos: tf.Tensor, adj_matr: tf.Tensor, shuffle: bool = True
) -> tf.Tensor:
    """Returns a dataset of node combinations that have equal amounts of
    adjacent and non-adjacent node pairs. The dataset elements are
    in alternating order: adj > not adj > adj > not adj > ..."""
    adjacencies = [(pair, get_combo_adjacency(pair, adj_matr)) for pair in all_combos]

    # categorise according to adjacency
    adj_combos = [pair for (pair, adj) in adjacencies if adj == 1]
    not_adj_combos = [pair for (pair, adj) in adjacencies if adj == 0]

    if shuffle is True:
        np.random.shuffle(adj_combos)
        np.random.shuffle(not_adj_combos)

    # zip + nested list comprehension = interleaving while cutting off excess non_adj_nodes
    reduced_combos = [
        node
        for adj_and_non_adj in zip(adj_combos, not_adj_combos)
        for node in adj_and_non_adj
    ]

    return tf.stack(reduced_combos)


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
    node_pos: tf.Tensor,
    pos_list: tf.Tensor,
) -> tf.Tensor:
    coords = [node_pair_to_coords(p, pos_list) for p in batch_combo]
    node_pair_imgs = [
        rc_to_node_combo_img(rc1, rc2, skel_img.shape) for (rc1, rc2) in coords
    ]
    combo_imgs = [
        np.stack([skel_img, np_im, node_pos], axis=-1).astype(np.uint8)
        for np_im in node_pair_imgs
    ]

    return tf.stack(combo_imgs)


def get_combo_adjacency(pair: tf.Tensor, adj_matr: tf.Tensor) -> tf.Tensor:
    n1, n2 = pair[0], pair[1]
    return adj_matr[n1, n2]


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


def rebatch(x: tf.data.Dataset, batch_size: int) -> tf.Tensor:
    return (
        x.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        .take(1)
        .get_single_element()
    )
