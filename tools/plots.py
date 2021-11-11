import cv2
import matplotlib.pyplot as plt
import numpy as np

from tools.image import normalize_mask, colour_codes, draw_circles


def plot_img(img, ax=None, cmap=None):
    if not ax:
        plt.imshow(img, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    else:
        ax.imshow(img, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_sample_from_train_generator(training_generator, batch_id=0):
    batch_size = training_generator.batch_size
    x, y = training_generator.__getitem__(batch_id)

    plt.figure()
    fig, axes = plt.subplots(batch_size, 1 + training_generator.n_classes)

    for i in range(batch_size):
        input_img = x[i, :, :, :]
        filtered_img = y[0][i, :, :, 0]
        skeletonised_img = y[1][i, :, :, 0]

        plot_img(input_img, axes[i, 0])

        plot_img(filtered_img, axes[i, 1], cmap='gray')
        plot_img(skeletonised_img, axes[i, 2], cmap='gray')

    axes[0, 0].set_title('Input')
    axes[0, 1].set_title('Output 1')
    axes[0, 2].set_title('Output 2')

    plt.show()

    # from tools.utilz_graph import tensor_2_adjmatrix, tensor_2_image_and_pos
    #
    # Tensor_sample = Tensor[batch_nr,]
    # img, pos = tensor_2_image_and_pos(Tensor_sample)
    # adj_matrix = tensor_2_adjmatrix(adj_vector = adj_vector[batch_nr, :],
    #                                 networksize = training_generator.max_node_dim,
    #                                 nr_nodes = len(pos))
    #
    # node_img = plot_nodes_on_img(img, pos, node_thick=6)
    # fig = plot_graph_on_img(img, pos, adj_matrix)


def plot_validation_results(validation_generator, results, batch_id=0):
    batch_size = validation_generator.batch_size
    x, y_true = validation_generator.__getitem__(batch_id)

    plt.figure()
    fig, axes = plt.subplots(batch_size, 1 + 2 * validation_generator.n_classes)

    for i in range(batch_size):
        input_img = x[i, :, :, :]
        filtered_img_true = y_true[0][i, :, :, 0]
        skeletonised_img_true = y_true[1][i, :, :, 0]

        filtered_img_res = (results[0][i] * 255).astype('uint8')

        binary_img = normalize_mask(results[1][i])
        skeletonised_img_res = binary_img.astype('uint8')

        plot_img(input_img, axes[i, 0])

        plot_img(filtered_img_true, axes[i, 1], cmap='gray')
        plot_img(skeletonised_img_true, axes[i, 3], cmap='gray')

        plot_img(filtered_img_res, axes[i, 2], cmap='gray')
        plot_img(skeletonised_img_res, axes[i, 4], cmap='gray')

    axes[0, 0].set_title('Input')
    axes[0, 1].set_title('Filtered')
    axes[0, 3].set_title('Skeletonised')

    plt.show()


def plot_sample(images: dict, title=''):
    for i, (label, img) in enumerate(images.items()):
        plt.subplot(230 + i + 1)

        cmap = 'gray' if img.shape[2] == 1 else None
        plot_img(img, cmap=cmap)
        plt.title(label)
    plt.suptitle(title)
    plt.show()


def plot_generated_images(iterator, title: str = '', cmap=None):
    base_imgs = []
    for i in range(4):
        plt.subplot(220 + 1 + i)
        batch = iterator.next()
        image = batch[0].astype('uint8')

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
            out_matrix = np.round(batch[0].squeeze()).astype('uint8')

            base_img = cv2.cvtColor(base_imgs[i], cv2.COLOR_GRAY2BGR).astype(np.uint8)
            out_image = draw_circles(base_img, out_matrix, colour_codes[k])
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
            plot_img(out_image)

        plt.suptitle(k)
        plt.show()
