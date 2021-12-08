from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from tools.image import classifier_preview, classify, colour_enums, draw_circles


def plot_img(img: np.ndarray, ax=None, cmap: Optional[str] = None):
    if not ax:
        plt.imshow(img, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    else:
        ax.imshow(img, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_training_sample(dataset, batch_id: int = 0, rows: int = 3):
    """
    Plots the training data (inputs and labels) of a batch.
    :param dataset: dataset containing training data
    :param batch_id: batch to use
    :param rows: maximum number of data points to plot
    :return:
    """
    skel_img, node_attrs = dataset[batch_id]

    assert rows <= dataset.batch_size

    for row in range(rows):
        plot_sample(skel_img, node_attrs, row, rows)

    plt.show()


def plot_sample(x: np.ndarray, y: np.ndarray, row: int = 0, rows: int = 0):
    output_names = ["node_pos", "degrees", "node_types"]

    def _subplot_id(row, col):
        """This function uses 0-indexing."""
        return col + 1 + 4 * row

    # set titles
    if row == 0:
        for col, t in enumerate(["skel"] + output_names):
            plt.subplot(rows, 4, _subplot_id(row, col))
            plt.title(t)

    # input
    plt.subplot(rows, 4, _subplot_id(row, 0))
    input_img = np.float32(x[row, :, :, :])
    plot_img(input_img, cmap="gray")

    # outputs
    output_matrices = {attr: y[i][row, :, :, 0] for i, attr in enumerate(output_names)}
    output_images = classifier_preview(output_matrices, input_img * 255)

    for col, attr in enumerate(output_names):
        plt.subplot(rows, 4, _subplot_id(row, col + 1))
        plot_img(output_images[attr])


def plot_augmented(x_iter, y_iter):
    base_imgs = plot_augmented_inputs(x_iter, "skeleton", cmap="gray")
    plot_augmented_outputs(y_iter, base_imgs)


def plot_augmented_inputs(iterator, title: str = "", cmap=None):
    base_imgs = []
    for i in range(4):
        plt.subplot(220 + 1 + i)
        batch = iterator.next()
        image = batch[0]

        base_imgs.append(image)
        plot_img(image, cmap=cmap)

    plt.suptitle(title)
    plt.show()

    return base_imgs


def plot_augmented_outputs(output_iterators, base_imgs):
    """Scans the matrices for integer values (nodes)
    and plots markers according to the node attribute."""
    for k, output_iter in output_iterators.items():
        for i in range(4):
            plt.subplot(220 + 1 + i)
            batch = output_iter.next()
            out_matrix = np.round(batch[0].squeeze()).astype("uint8")

            base_img = cv2.cvtColor(base_imgs[i] * 255, cv2.COLOR_GRAY2BGR).astype(
                np.uint8
            )
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


def show_predictions(model, dataset, batch=0, filepath=None):
    input_images, masks = dataset.get_batch_data(batch)
    pred_masks = model.predict(input_images)

    b_id = 0
    input_image = input_images[b_id]

    gt_output_matrices = {
        attr: masks[i][b_id]
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
