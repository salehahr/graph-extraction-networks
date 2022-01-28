from typing import Dict, List, Optional

import cv2
import networkx as nx
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from tools.data import pos_list_from_image
from tools.image import (
    classifier_preview,
    classify,
    colour_enums,
    draw_circles,
    get_rgb,
)


def plot_img(img: np.ndarray, ax=None, cmap: Optional[str] = None):
    if not ax:
        plt.imshow(img, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    else:
        ax.imshow(img, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_bgr_img(img, title=""):
    n_channels = img.shape[2] if len(img.shape) >= 3 else 1
    cmap = "gray" if n_channels == 1 else None

    image = get_rgb(img)

    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)


def plot_training_sample(
    data_generator, network: int, step_num: int = 0, rows: int = 3
):
    """
    Plots the training data (inputs and labels) of a batch.
    :param dataset: dataset containing training data
    :param step_num: which epoch step
    :param rows: maximum number of data points to plot
    :return:
    """
    b_inputs, b_outputs = data_generator[step_num]

    rows = data_generator.batch_size if rows > data_generator.batch_size else rows

    plot_fcn = plot_sample if network == 2 else plot_sample_third_network

    for row in range(rows):
        plt.figure(0)
        plot_fcn(b_inputs, b_outputs, row, rows)

    plt.show()


def plot_sample(x: tf.Tensor, y: tuple, row: int = 0, rows: int = 0):
    input_names = ["skel"]
    output_names = ["node_pos", "degrees", "node_types"]
    data_names = input_names + output_names

    set_plot_title(data_names, row, rows)

    # input
    plt.subplot(rows, 4, get_subplot_id(row, 0))
    input_img = np.float32(x.numpy()[row, :, :, :])
    plot_img(input_img, cmap="gray")

    # outputs
    output_matrices = {
        attr: y[i].numpy()[row, :, :, 0] for i, attr in enumerate(output_names)
    }
    output_images = classifier_preview(output_matrices, input_img * 255)

    for col, attr in enumerate(output_names):
        plt.subplot(rows, 4, get_subplot_id(row, col + 1))
        plot_img(output_images[attr])


def plot_sample_third_network(x, y, row: int, rows: int):
    input_names = ["skel", "node_pos", "degrees"]
    output_names = ["adj_matr"]
    data_names = input_names + output_names

    set_plot_title(data_names, row, rows)

    # skel_image
    plt.subplot(rows, 4, get_subplot_id(row, 0))
    skel_img = np.float32(x[0].numpy()[row, :, :, :])
    plot_img(skel_img, cmap="gray")

    # node_attributes
    output_matrices = {
        attr: x[i].numpy()[row, :, :, 0]
        for i, attr in enumerate(input_names[1:], start=1)
    }
    output_images = classifier_preview(output_matrices, skel_img * 255)

    for col, attr in enumerate(input_names[1:], start=1):
        plt.subplot(rows, 4, get_subplot_id(row, col))
        plot_img(output_images[attr])

    # adjacency
    plt.subplot(rows, 4, get_subplot_id(row, 3))
    pos_list_xy = pos_list_from_image(output_matrices["node_pos"])
    adj_matr = y[row].numpy()

    plot_adj_matr(skel_img, pos_list_xy, adj_matr)

    # big figure adj
    plt.figure(row + 1)
    plot_adj_matr(skel_img, pos_list_xy, adj_matr)


def set_plot_title(data_names: list, row: int, rows: int):
    if row == 0:
        for col, t in enumerate(data_names):
            plt.subplot(rows, 4, get_subplot_id(row, col))
            plt.title(t)


def get_subplot_id(row, col):
    """This function uses 0-indexing."""
    return col + 1 + 4 * row


def plot_adj_matr(img_skel: np.ndarray, pos: list, adjacency: np.ndarray) -> None:
    """
    Function for checking if the adjacency matrix matches the image
    by overlaying the graph over the skeletonised image.
    :param img_skel: skeletonised image
    :param pos: list of position coordinates of the graph nodes
    :param adjacency: adjacency matrix of the graph
    """
    img = img_skel.copy()

    img_height = img.shape[0]
    pos_dict = {i: [x, img_height - y] for i, [x, y] in enumerate(pos)}

    adjacency = adjacency.squeeze() if adjacency.ndim == 3 else adjacency
    graph = nx.from_numpy_array(adjacency)
    nx.set_node_attributes(graph, pos_dict, "pos")

    y_lim, x_lim = img.shape if img.ndim == 2 else img.shape[:-1]
    extent = 0, x_lim, 0, y_lim

    plt.imshow(img, extent=extent, interpolation="nearest", cmap="gray")
    nx.draw(graph, pos=pos_dict, node_size=2, node_color="r", edge_color="g", width=1)


def plot_augmented(x: List[np.ndarray], y: Dict[str, np.ndarray]):
    base_imgs = plot_augmented_inputs(x, "skeleton", cmap="gray")
    plot_augmented_outputs(y, base_imgs)


def plot_augmented_inputs(
    b_imgs: List[np.ndarray], title: str = "", cmap=None
) -> List[np.ndarray]:
    base_imgs = [batch[0] for batch in b_imgs]

    for i, img in enumerate(base_imgs):
        plt.subplot(220 + 1 + i)
        plot_img(img, cmap=cmap)

    plt.suptitle(title)
    plt.show()

    return base_imgs


def plot_augmented_outputs(
    graph_data: Dict[str, np.ndarray], base_imgs: List[np.ndarray]
):
    """Scans the matrices for integer values (nodes)
    and plots markers according to the node attribute."""

    for k, data in graph_data.items():
        for i, img in enumerate(data):
            plt.subplot(220 + 1 + i)

            out_matrix = np.round(img[0].squeeze()).astype("uint8")
            base_img = cv2.cvtColor(base_imgs[i] * 255, cv2.COLOR_GRAY2BGR).astype(
                np.uint8
            )

            if k == "adj_matr":
                node_pos = graph_data["node_pos"][i][0]
                node_pos = np.round(node_pos.squeeze().astype("uint8"))
                plot_adj_matr(
                    base_img,
                    pos_list_from_image(node_pos),
                    out_matrix,
                )
            else:
                out_image = draw_circles(base_img, out_matrix, colour_enums[k])
                out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
                plot_img(out_image)

        plt.suptitle(k)
        plt.show()


def display_single_output(display_list: list, big_title: str):
    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        image = display_list[i]
        if image is None:
            continue

        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(image))
        plt.axis("off")

    plt.suptitle(big_title)
    plt.show()


def show_predictions(
    model: tf.keras.models.Model,
    data_generator,
    batch: int = 0,
    filepath: str = None,
):
    input_images, masks = data_generator[batch]
    pred_masks = model.predict(input_images)

    b_id = 0
    input_image = input_images[b_id].numpy()

    gt_output_matrices = {
        attr: masks[i][b_id].numpy()
        for i, attr in enumerate(["node_pos", "degrees", "node_types"])
    }
    gt_output_images = classifier_preview(gt_output_matrices, input_image * 255)

    pred_output_matrices = {
        attr: classify(pred_masks[i][b_id])[0]
        for i, attr in enumerate(["node_pos", "degrees", "node_types"])
    }
    pred_output_images = classifier_preview(pred_output_matrices, input_image * 255)

    plt.subplot(331)
    plot_img(input_image, cmap="gray")
    plt.xlabel("Input")

    for i, attr in enumerate(["node_pos", "degrees", "node_types"]):
        plt.subplot(330 + 3 * (i + 1))
        if i == 0:
            plt.title("Truth")
        plot_img(gt_output_images[attr])
        plt.xlabel(attr)

        plt.subplot(330 + 3 * (i + 1) - 1)
        if i == 0:
            plt.title("Predicted")
        plot_img(pred_output_images[attr])
        plt.xlabel(attr)

    if filepath:
        plt.savefig(filepath)

    plt.show()
