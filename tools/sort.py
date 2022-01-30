from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf


def get_sort_indices(xy_unsorted: tf.Tensor) -> tf.Tensor:
    """
    Returns the sort indices when sorting by y/rows first, then x/cols.
    """
    idx_y_sorted = tf.argsort(xy_unsorted[:, 1])
    y_sorted = tf.gather(xy_unsorted, idx_y_sorted)

    idx_x_sorted = tf.argsort(y_sorted[:, 0])
    idx_xy_sorted = tf.gather(idx_y_sorted, idx_x_sorted)

    return idx_xy_sorted


def sort_list_of_nodes(unsorted: Union[List[List[int]], np.ndarray]) -> np.ndarray:
    """
    Returns the sorted nodes.
    :param unsorted: unsorted nodes
    :return: sorted nodes
    """
    _, sorted_nodes = sort_nodes(unsorted)
    return sorted_nodes


def sort_nodes(
    unsorted: Union[List[List[int]], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the new indices relative to the old list, as well as the sorted list.
    :param unsorted: unsorted nodes
    :return: sort indices, sorted nodes
    """
    sorted_tuple = sorted(enumerate(unsorted), key=lambda x: [x[1][0], x[1][1]])
    indices, sorted_nodes = zip(*sorted_tuple)
    return np.array(indices), np.array(sorted_nodes)
