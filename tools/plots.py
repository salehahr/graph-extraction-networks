from __future__ import annotations

from time import time
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import cv2
import networkx as nx
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from tools.colours import BGR_GREEN
from tools.data import pos_list_from_image
from tools.files import create_folder
from tools.image import classifier_preview, colour_enums, draw_circles, get_rgb
from tools.NetworkType import NetworkType
from tools.node_classifiers import NodeDegrees
from tools.postprocessing import classify

if TYPE_CHECKING:
    from tools import EdgeDG, EdgeDGSingle


def plot_imgs(imgs: Union[List[np.ndarray], tf.Tensor], show: bool = True):
    for im in imgs:
        plot_img(im)

        if show:
            plt.show()


def plot_img(img: np.ndarray, ax=None, cmap: Optional[str] = None):
    """Displays image without ax ticks."""
    if not ax:
        plt.imshow(img, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    else:
        ax.imshow(img, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_bgr_img(img, title="", show: bool = False):
    n_channels = img.shape[2] if len(img.shape) >= 3 else 1
    cmap = "gray" if n_channels == 1 else None

    image = get_rgb(img)

    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

    if show:
        plt.show()


def plot_training_sample(
    data_generator, network: NetworkType, step_num: int = 0, rows: int = 3, **kwargs
):
    """
    Plots the training data (inputs and labels) of a batch.
    :param data_generator: training data generator
    :param network: network id
    :param step_num: which epoch step
    :param rows: maximum number of data points to plot
    :return:
    """
    rows = min(data_generator.batch_size, rows)

    if network == NetworkType.NODES_NN:
        plot_fcn = plot_sample_nodes_nn
    elif network == NetworkType.ADJ_MATR_NN:
        plot_fcn = plot_sample_adj_nn
    elif network == NetworkType.EDGE_NN:
        plot_fcn = plot_sample_edge_nn
    else:
        raise Exception

    if network == NetworkType.EDGE_NN:
        plot_fcn(data_generator, step_num, 0, rows, **kwargs)
    else:
        for row in range(rows):
            plt.figure(0)
            plot_fcn(data_generator, step_num, row, rows, **kwargs)

    plt.show()


def plot_sample_nodes_nn(
    data_generator, step_num: int, row: int = 0, rows: int = 0, **kwargs
):
    plot_small = kwargs.get("small_section")

    input_names = ["skel"]
    output_names = ["node_pos", "degrees", "node_types"]
    data_names = input_names + output_names
    num_cols = len(data_names)

    set_plot_title(data_names, row, rows)

    b_skel_img, b_y = data_generator[step_num]
    if plot_small:
        b_skel_img = b_skel_img[:, 100:200, 100:200, :]
        b_y = tuple(y[:, 100:200, 100:200, :] for y in b_y)

    # input
    plt.subplot(rows, 4, get_subplot_id(row, 0, num_cols=num_cols))
    input_img = np.float32(b_skel_img[row].numpy())

    plot_img(input_img, cmap="gray")

    # outputs
    output_matrices = {
        attr: b_y[i].numpy()[row, :, :, 0] for i, attr in enumerate(output_names)
    }
    output_images = classifier_preview(output_matrices, input_img * 255)

    for col, attr in enumerate(output_names):
        plt.subplot(rows, 4, get_subplot_id(row, col + 1, num_cols=num_cols))
        plot_img(output_images[attr])


def plot_sample_adj_nn(data_generator, step_num: int, row: int, rows: int, **kwargs):
    input_names = ["skel", "node_pos", "degrees"]
    output_names = ["adj_matr"]
    data_names = input_names + output_names
    num_cols = len(data_names)

    set_plot_title(data_names, row, rows)

    b_x, b_y = data_generator[step_num]

    # skel_image
    plt.subplot(rows, 4, get_subplot_id(row, 0, num_cols=num_cols))
    skel_img = np.float32(b_x[0].numpy()[row, :, :, :])
    plot_img(skel_img, cmap="gray")

    # node_attributes
    output_matrices = {
        attr: b_x[i].numpy()[row, :, :, 0]
        for i, attr in enumerate(input_names[1:], start=1)
    }
    output_images = classifier_preview(output_matrices, skel_img * 255)

    for col, attr in enumerate(input_names[1:], start=1):
        plt.subplot(rows, 4, get_subplot_id(row, col, num_cols=num_cols))
        plot_img(output_images[attr])

    # adjacency matrix
    plt.subplot(rows, 4, get_subplot_id(row, 3, num_cols=num_cols))
    pos_list_xy = pos_list_from_image(output_matrices["node_pos"])
    adj_matr = b_y[row].numpy()

    plot_adj_matr(skel_img, pos_list_xy, adj_matr)

    # big figure adj
    plt.figure(row + 1)
    plot_adj_matr(skel_img, pos_list_xy, adj_matr)


def plot_sample_edge_nn(
    data_generator: Union[EdgeDGSingle, EdgeDG],
    step_num: int,
    row: int,
    num_rows: int,
):
    num_cols = 2

    assert data_generator.with_path is True
    combo_img, (adjacencies, paths) = data_generator[step_num]
    adjacencies = adjacencies.numpy().squeeze()
    paths = [p.numpy() for p in paths]

    num_imgs = data_generator.images_in_batch
    num_combos = data_generator.node_pairs_image

    # repartition data
    adjacencies = partition_data(adjacencies, num_imgs)
    paths = partition_data(paths, num_imgs)

    for i in range(num_imgs):
        idx = i * num_combos

        skel_img = np.float32(combo_img[idx, ..., 0].numpy())
        node_pair_imgs = np.float32(combo_img[idx : idx + num_combos, ..., 1].numpy())

        im_adj = adjacencies[i]
        im_paths = paths[i]
        pairs_xy = [np.fliplr(np.argwhere(np_im)) for np_im in node_pair_imgs]

        plot_node_pairs_on_skel(skel_img, pairs_xy, show=True)
        set_plot_title(["path", "path from DataGen"], row, num_rows)

        max_rows = min(num_rows, num_combos)
        for ii in range(max_rows):
            idx += 1

            # path
            rc1, rc2 = np.fliplr(pairs_xy[ii])
            rows = np.sort([rc1[0], rc2[0]])
            cols = np.sort([rc1[1], rc2[1]])

            img_section = skel_img[rows[0] : rows[1] + 1, cols[0] : cols[1] + 1]
            plt.subplot(num_rows, num_cols, get_subplot_id(ii, 0, num_cols))
            plot_img(img_section, cmap="gray")
            plt.xlabel(f"RC {rc1} - {rc2}")

            plt.subplot(num_rows, num_cols, get_subplot_id(ii, 1, num_cols))
            plot_img(im_paths[ii], cmap="gray")
            plt.xlabel(f"adj {im_adj[ii]}")

        plt.show()


def partition_data(data: Union[list, np.ndarray], num_partitions: int) -> List:
    return np.array_split(data, num_partitions)


def plot_node_pairs_on_skel(skel_img, pairs_xy: list, show: bool = False) -> np.ndarray:
    # image conversion
    if skel_img.dtype is not tf.uint8:
        skel_img = tf.image.convert_image_dtype(skel_img, tf.uint8).numpy()
    else:
        skel_img = skel_img.numpy()
    skel_img = skel_img.squeeze()

    rgb_img = np.repeat(np.expand_dims(skel_img, axis=-1), 3, axis=2)
    marker_size = 3

    for i, (xy1, xy2) in enumerate(pairs_xy, start=1):
        colour = BGR_GREEN if len(pairs_xy) == 1 else NodeDegrees(i).colour
        cv2.circle(rgb_img, tuple(xy1), marker_size, colour, -1)
        cv2.circle(rgb_img, tuple(xy2), marker_size, colour, -1)

    if show:
        plot_bgr_img(rgb_img, title="1: white, 2: green, 3: red", show=show)

    return rgb_img


def set_plot_title(data_names: list, row: int, rows: int):
    num_cols = len(data_names)
    if row == 0:
        for col, t in enumerate(data_names):
            plt.subplot(rows, num_cols, get_subplot_id(row, col, num_cols))
            plt.title(t)


def get_subplot_id(row, col, num_cols):
    """This function uses 0-indexing."""
    return col + 1 + num_cols * row


def lighten_skel_img(img_skel: np.ndarray, black_to_grey: float = 0.7) -> np.ndarray:
    img = img_skel.copy() / 2

    ids = np.argwhere(img_skel == 0)
    img[ids[:, 0], ids[:, 1]] = black_to_grey
    ids = np.argwhere(img_skel == 1)
    img[ids[:, 0], ids[:, 1]] = np.mean([1 - black_to_grey, 1 / black_to_grey])

    cmap = plt.get_cmap("gray")
    return np.squeeze(cmap(img))


def plot_adj_matr(
    img_skel: np.ndarray,
    pos: np.ndarray,
    adjacency: np.ndarray,
    show: bool = True,
    with_numbers: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    Function for checking if the adjacency matrix matches the image
    by overlaying the graph over the skeletonised image.
    :param img_skel: skeletonised image
    :param pos: list of position coordinates of the graph nodes
    :param adjacency: adjacency matrix of the graph
    :param show: whether to display plot or not
    :param with_numbers: whether to display node IDs in plot
    :param title: title of the plot
    """
    img = lighten_skel_img(img_skel)

    img_height = img.shape[0]
    pos_dict = {i: [x, img_height - y] for i, [x, y] in enumerate(pos)}

    adjacency = adjacency.squeeze() if adjacency.ndim == 3 else adjacency
    graph = nx.from_numpy_array(adjacency)
    nx.set_node_attributes(graph, pos_dict, "pos")

    y_lim, x_lim = img.shape if img.ndim == 2 else img.shape[0:2]
    extent = 0, x_lim, 0, y_lim

    plt.imshow(img, extent=extent, interpolation="nearest", cmap="gray")
    plt.title(title, loc="left")

    # annotate
    if with_numbers:
        for i, xy in pos_dict.items():
            plt.annotate(
                f"{i}",
                xy=xy,
                xytext=(-3, 3),
                textcoords="offset points",
                horizontalalignment="right",
                verticalalignment="bottom",
                fontsize="xx-small",
                color="white",
                # bbox=dict(
                #     boxstyle="circle", ec="none", fc="navajowhite", alpha=0.5, pad=0.1
                # ),
            )

    # graph
    nx.draw(
        graph,
        pos=pos_dict,
        node_size=8,
        node_color="navajowhite",
        edge_color="hotpink",
        width=1.5,
    )

    if show:
        plt.show()


def plot_augmented(x: List[np.ndarray], y: Dict[str, List[np.ndarray]]):
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


def _get_prediction_images(
    model: tf.keras.models.Model,
    data_generator,
    batch: int = 0,
    id_in_batch: int = 0,
    with_time: bool = False,
    with_filepath: bool = False,
):
    input_images, masks, filepaths = data_generator[batch]
    filepath = filepaths[id_in_batch].replace(" ", "").replace(":", "-")

    num_iters = 3 if with_time else 1
    for i in range(num_iters):
        t_start = time()
        pred_masks = model.predict(input_images)
        t_elapsed = time() - t_start

    input_im = input_images[id_in_batch].numpy()
    input_im = lighten_skel_img(input_im)[:, :, :3]

    gt_output_matrices = {
        attr: masks[i][id_in_batch].numpy()
        for i, attr in enumerate(["node_pos", "degrees", "node_types"])
    }
    gt_imgs = classifier_preview(gt_output_matrices, input_im * 255)

    pred_output_matrices = {
        attr: classify(pred_masks[i][id_in_batch])[0]
        for i, attr in enumerate(["node_pos", "degrees", "node_types"])
    }
    pred_imgs = classifier_preview(pred_output_matrices, input_im * 255)

    # # get predictions with misclassification(s)
    # for attr, img in gt_output_matrices.items():
    # if not np.array_equal(img, pred_output_matrices[attr]):
    #     diff = img - pred_output_matrices[attr]
    #     diff_idx = np.where(diff != 0)

    if with_time:
        if with_filepath:
            return input_im, gt_imgs, pred_imgs, t_elapsed, filepath
        else:
            return input_im, gt_imgs, pred_imgs, t_elapsed
    else:
        if with_filepath:
            return input_im, gt_imgs, pred_imgs, filepath
        else:
            return input_im, gt_imgs, pred_imgs


def save_prediction_images(
    model: tf.keras.models.Model,
    data_generator,
    batch: int = 0,
    id_in_batch: int = 0,
    prefix: str = None,
):
    (
        input_image,
        gt_output_images,
        pred_output_images,
        t_elapsed,
        filepath,
    ) = _get_prediction_images(
        model, data_generator, batch, with_time=True, with_filepath=True
    )
    print(f"t_elapsed {t_elapsed} s")

    folder = f"data/nodesnn"
    model_folder = f"{folder}/{prefix}"
    batch_prefix = f"b{batch:03d}-{id_in_batch:04d}"
    input_folder = f"{folder}/inputs"

    # make folders if not existing
    create_folder(input_folder)

    for attr in gt_output_images.keys():
        gt_folder = f"{folder}/{attr}"
        pred_folder = f"{model_folder}/{attr}"

        create_folder(gt_folder)
        create_folder(pred_folder)

    filename = f"{batch_prefix}-{filepath}.png"

    # input
    input_image = (input_image * 255).astype(np.uint8)
    cv2.imwrite(f"{input_folder}/{filename}", input_image)

    for attr, img in gt_output_images.items():
        cv2.imwrite(
            f"{folder}/{attr}/{filename}",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            f"{model_folder}/{attr}/{filename}",
            cv2.cvtColor(pred_output_images[attr], cv2.COLOR_RGB2BGR),
        )


def show_predictions(
    model: tf.keras.models.Model,
    data_generator,
    batch: int = 0,
    filepath: str = None,
):
    input_image, gt_output_images, pred_output_images = _get_prediction_images(
        model, data_generator, batch
    )

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
