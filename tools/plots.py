from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tools.image import classifier_preview, colour_enums, draw_circles, normalize_mask


def plot_img(img: np.ndarray, ax=None, cmap: Optional[str] = None):
    if not ax:
        plt.imshow(img, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    else:
        ax.imshow(img, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_training_sample(dataset, batch_id: int = 0):
    batch_size = dataset.batch_size
    skel_img, node_attrs = dataset[batch_id]

    plt.figure()
    fig, axes = plt.subplots(batch_size, 1 + dataset.output_channels)

    for b in range(batch_size):
        input_img = np.float32(skel_img[b, :, :, :])
        plot_img(input_img, axes[b, 0], cmap="gray")

        output_matrices = {
            attr: node_attrs[i][b, :, :, 0]
            for i, attr in enumerate(["node_pos", "degrees", "node_types"])
        }

        output_images = classifier_preview(output_matrices, input_img * 255)
        plot_img(
            output_images["node_pos"],
            axes[b, 1],
        )
        plot_img(output_images["degrees"], axes[b, 2])
        plot_img(output_images["node_types"], axes[b, 3])

    axes[0, 0].set_title("Skel")
    axes[0, 1].set_title("Node pos")
    axes[0, 2].set_title("Node degrees")
    axes[0, 3].set_title("Node types")

    plt.show()


def plot_validation_results(validation_generator, results, batch_id=0):
    batch_size = validation_generator.batch_size
    x, y_true = validation_generator.__getitem__(batch_id)

    plt.figure()
    fig, axes = plt.subplots(batch_size, 1 + 2 * validation_generator.output_channels)

    for i in range(batch_size):
        input_img = x[i, :, :, :]
        filtered_img_true = y_true[0][i, :, :, 0]
        skeletonised_img_true = y_true[1][i, :, :, 0]

        filtered_img_res = (results[0][i] * 255).astype("uint8")

        binary_img = normalize_mask(results[1][i])
        skeletonised_img_res = binary_img.astype("uint8")

        plot_img(input_img, axes[i, 0])

        plot_img(filtered_img_true, axes[i, 1], cmap="gray")
        plot_img(skeletonised_img_true, axes[i, 3], cmap="gray")

        plot_img(filtered_img_res, axes[i, 2], cmap="gray")
        plot_img(skeletonised_img_res, axes[i, 4], cmap="gray")

    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Filtered")
    axes[0, 3].set_title("Skeletonised")

    plt.show()


def plot_sample(images: dict, title=""):
    for i, (label, img) in enumerate(images.items()):
        plt.subplot(230 + i + 1)

        cmap = "gray" if img.shape[2] == 1 else None
        plot_img(img, cmap=cmap)
        plt.title(label)
    plt.suptitle(title)
    plt.show()


def plot_generated_images(iterator, title: str = "", cmap=None):
    base_imgs = []
    for i in range(4):
        plt.subplot(220 + 1 + i)
        batch = iterator.next()
        image = batch[0].astype("uint8")

        base_imgs.append(image)
        plot_img(image, cmap=cmap)

    plt.suptitle(title)
    plt.show()

    return base_imgs


def plot_classifier_images(output_iterators, base_imgs):
    for k, output_iter in output_iterators.items():
        for i in range(4):
            plt.subplot(220 + 1 + i)
            batch = output_iter.next()
            out_matrix = np.round(batch[0].squeeze()).astype("uint8")

            base_img = cv2.cvtColor(base_imgs[i], cv2.COLOR_GRAY2BGR).astype(np.uint8)
            out_image = draw_circles(base_img, out_matrix, colour_enums[k])
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
            plot_img(out_image)

        plt.suptitle(k)
        plt.show()
