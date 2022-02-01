import os

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
