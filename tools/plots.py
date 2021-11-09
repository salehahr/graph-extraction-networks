import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tools.image import normalize_mask


def plot_graph_on_img(image: np.ndarray, pos: np.ndarray, adjacency: np.ndarray):
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    adjacency_matrix = np.uint8(adjacency.copy())
    positions = pos.copy()
    pos_list = []
    for i in range(len(positions)):
        pos_list.append([positions[i][0], img.shape[0] - positions[i][1]])
    p = dict(enumerate(pos_list, 0))

    graph = nx.from_numpy_matrix(adjacency_matrix)
    nx.set_node_attributes(graph, p, 'pos')

    y_lim, x_lim = img.shape[:-1]
    extent = 0, x_lim, 0, y_lim

    fig = plt.figure(frameon=False, figsize=(20, 20))
    plt.imshow(img, extent=extent, interpolation='nearest')
    nx.draw(graph, pos=p, node_size=50, edge_color='g', width=3, node_color='r')

    plt.show()

    return fig


def plot_nodes_on_img(image: np.ndarray, pos: np.ndarray, node_thick: int):
    img = image.copy()
    print(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # positions = pos.astype(int)
    positions = pos
    for i in range(len(positions)):
        cv2.circle(img, (positions[i][0], positions[i][1]), 0, (255, 0, 0), node_thick)
    y_lim, x_lim = img.shape[:-1]
    extent = 0, x_lim, 0, y_lim
    plt.figure(frameon=False, figsize=(20, 20))
    plt.imshow(img, extent=extent, interpolation='nearest')
    plt.show()
    return img


def get_rgb_from_bgr(img):
    """ Gets RGB image for matplotlib plots. """
    if img.max() < 1:
        img = (img * 255).astype('uint8')
    else:
        img = img.astype('uint8')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
