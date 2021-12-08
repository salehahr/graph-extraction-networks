from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf

from .node_classifiers import NodeDegrees, NodePositions, NodeTypes

marker_size = 3


colour_enums = {
    "node_pos": NodePositions,
    "degrees": NodeDegrees,
    "node_types": NodeTypes,
}


def classify(mask: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Returns mask with integer classes."""
    is_binary = mask.shape[-1] <= 2

    if is_binary:
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    else:
        mask = tf.argmax(mask, axis=-1)
        mask = mask[..., tf.newaxis].numpy()

    return mask, is_binary


def generate_outputs(graph, dim: int) -> dict:
    """
    Generates output matrices of the graph's node attributes.
    """
    matr_node_pos = np.zeros((dim, dim, 1)).astype(np.uint8)
    matr_node_degrees = np.zeros((dim, dim, 1)).astype(np.uint8)
    matr_node_types = np.zeros((dim, dim, 1)).astype(np.uint8)

    for i, (col, row) in enumerate(graph.positions):
        matr_node_pos[row][col] = 1
        matr_node_degrees[row][col] = graph.num_node_neighbours[i]
        matr_node_types[row][col] = graph.node_types[i]

    return {
        "node_pos": matr_node_pos,
        "degrees": matr_node_degrees,
        "node_types": matr_node_types,
    }


def classifier_preview(output_matrices: dict, img_skel: np.ndarray) -> dict:
    """
    Serves as a visual test to check the correctness of the output matrices.
    Takes the classifier matrices as an input and generates for each matrix
    the corresponding visualisation.
    """
    base_img = cv2.cvtColor(img_skel, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    data_dict = {
        attr: {"matrix": matr.squeeze(), "colours": colour_enums[attr]}
        for attr, matr in output_matrices.items()
    }

    debug_images = {}
    for attr, v in data_dict.items():
        img = draw_circles(base_img, v["matrix"], v["colours"])
        debug_images[attr] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return debug_images


def draw_circles(
    base_img: np.ndarray, classifier_matrix: np.ndarray, colours: list
) -> np.ndarray:
    """
    Draws circles on the image colour coded according to the unique values
    in the classifier matrix.
    Returns BGR image.
    """
    img = base_img.copy()
    unique_vals = np.unique(classifier_matrix)[1:]

    for val in unique_vals:
        positions = np.argwhere(classifier_matrix == val)
        for (y, x) in positions:
            cv2.circle(img, (x, y), marker_size, colours(val).colour, -1)

    return img
