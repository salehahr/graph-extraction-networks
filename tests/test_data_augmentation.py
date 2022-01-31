import unittest

import networkx as nx
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from tools import Config, GraphExtractionDG, NodeExtractionDG, TestType
from tools.data import sort_nodes
from tools.plots import plot_augmented, plot_training_sample


class TestDataAugmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")

        cls.img_datagen = cls._init_old_data_augmenter()
        cls.aug_seed = 1

        cls.network_num, cls.training_data = cls._gen_training_data()

    @classmethod
    def _init_old_data_augmenter(cls):
        data_gen_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
        )

        return ImageDataGenerator(**data_gen_args)

    @classmethod
    def _gen_training_data(cls):
        """Generates training data based on network ID."""
        network_num = 2
        training_data = NodeExtractionDG(
            cls.config,
            cls.config.network.node_extraction,
            TestType.TRAINING,
            augmented=False,
        )

        return network_num, training_data

    def _get_samples(self):
        """Get first output of the first training batch."""
        x_batch, y_batch = self._get_first_batch()

        # expand dimensions for inputting to ImageDataGenerator.flow()
        x_first = np.expand_dims(x_batch[0].numpy(), axis=0)

        node_pos_first = np.expand_dims(y_batch[0][0].numpy(), axis=0)
        degrees_first = np.expand_dims(y_batch[1][0].numpy(), axis=0)
        node_types_first = np.expand_dims(y_batch[2][0].numpy(), axis=0)

        return x_first, node_pos_first, degrees_first, node_types_first

    def _get_first_batch(self):
        """Gets first training batch."""
        plot_training_sample(self.training_data, network=self.network_num, rows=1)

        # gets the first batch (>= 1 images in batch)
        return self.training_data[0]

    def test_old_data_aug(self):
        """
        For testing the method previously used for data augmentation.
        """
        skel_img, node_pos, degrees, node_types = self._get_samples()

        skel_img_aug = self._aug_imgs(skel_img)
        graph_aug = {
            "node_pos": self._aug_imgs(node_pos),
            "degrees": self._aug_imgs(degrees),
            "node_types": self._aug_imgs(node_types),
        }

        plot_augmented(skel_img_aug, graph_aug)

    def _aug_imgs(self, sample):
        img_iter = self.img_datagen.flow(sample, batch_size=1, seed=self.aug_seed)
        return [img_iter.next() for _ in range(4)]

    def test_data_aug(self):
        plot_training_sample(self.training_data, network=self.network_num, rows=3)

        self.training_data.augmented = True

        plot_training_sample(self.training_data, network=self.network_num, rows=3)


class TestThirdNetworkDataAugmentation(TestDataAugmentation):
    @classmethod
    def _gen_training_data(cls):
        """Generates training data based on network ID."""
        network_num = 3
        training_data = GraphExtractionDG(
            cls.config,
            cls.config.network.graph_extraction,
            TestType.TRAINING,
            augmented=False,
        )

        return network_num, training_data

    def _get_samples(self):
        """Get first output of the first training batch."""
        x_batch, y_batch = self._get_first_batch()
        b_skel_img, b_node_pos, b_degrees = x_batch

        # expand dimensions for inputting to ImageDataGenerator.flow()
        skel_img_first = np.expand_dims(b_skel_img[0].numpy(), axis=0)
        node_pos_first = np.expand_dims(b_node_pos[0].numpy(), axis=0)
        degrees_first = np.expand_dims(b_degrees[0].numpy(), axis=0)
        adj_matr_first = np.expand_dims(y_batch[0].numpy(), axis=0)

        return skel_img_first, node_pos_first, degrees_first, adj_matr_first

    def test_old_data_aug(self):
        """
        For testing the method previously used for data augmentation.
        """
        skel_img, node_pos, degrees, adj_matr = self._get_samples()

        skel_img_aug = self._aug_imgs(skel_img)
        graph_aug = {
            "node_pos": self._aug_imgs(node_pos),
            "degrees": self._aug_imgs(degrees),
            "adj_matr": self._aug_imgs(adj_matr),
        }

        plot_augmented(skel_img_aug, graph_aug)


class TestAdjacencyMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.min_x, cls.max_x = 1, 7
        cls.min_y, cls.max_y = 1, 6

        nodes_orig = [
            [1, 1],
            [1, 5],
            [2, 3],
            [3, 5],
            [4, 1],
            [5, 6],
            [6, 4],
            [7, 2],
            [7, 5],
        ]
        cls.edges = [
            [[1, 1], [2, 3]],
            [[1, 5], [2, 3]],
            [[2, 3], [4, 1]],
            [[2, 3], [5, 6]],
            [[3, 5], [6, 4]],
            [[7, 2], [7, 5]],
        ]
        cls.idx_orig, cls.nodes_orig = sort_nodes(nodes_orig)
        cls.num_nodes = len(nodes_orig)

        graph = cls.create_graph()
        cls.A = nx.to_numpy_array(graph).astype(np.uint8)

        print(f"Nodes:\n{cls.nodes_orig}")
        print(f"Original A:\n{cls.A}")

    @classmethod
    def create_graph(cls) -> nx.Graph:
        graph = nx.Graph()

        # define nodes with attribute position
        for i, xy in enumerate(cls.nodes_orig):
            graph.add_node(i, pos=tuple(xy))

        # define edges with attributes: length
        for edge in cls.edges:
            start_xy, end_xy = edge

            startidx = cls.nodes_orig.tolist().index(start_xy)
            endidx = cls.nodes_orig.tolist().index(end_xy)

            graph.add_edge(startidx, endidx)

        return graph

    def _transform_A(self, A: np.ndarray, idx_trans_sorted: np.ndarray) -> np.ndarray:
        I = np.identity(self.num_nodes, dtype=np.uint8)
        P = np.take(I, idx_trans_sorted, axis=0)
        return P @ A @ np.transpose(P)

    def _print_transformed_nodes(self, nodes, idx_new):
        for i, n, i_ts in zip(self.idx_orig, nodes, idx_new):
            print(f"{i}: {n} --> {i_ts}")

    def test_horz_flip(self):
        print("Horizontal flip")

        nodes_trans = self._apply_horz_flip(self.nodes_orig)
        idx_sorted, nodes_trans_sorted = sort_nodes(nodes_trans)
        idx_trans_sorted = np.take(self.idx_orig, idx_sorted)

        A_trans = self._transform_A(self.A, idx_trans_sorted)
        self._print_transformed_nodes(nodes_trans, idx_trans_sorted)
        print(f"Sorted:\n{nodes_trans_sorted}")
        print(A_trans)

        np.testing.assert_equal(
            A_trans,
            np.array(
                [
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                ]
            ),
        )

    def test_vert_flip(self):
        print("Vertical flip")

        nodes_trans = self._apply_vert_flip(self.nodes_orig)
        idx_sorted, nodes_trans_sorted = sort_nodes(nodes_trans)
        idx_trans_sorted = np.take(self.idx_orig, idx_sorted)

        A_trans = self._transform_A(self.A, idx_trans_sorted)
        self._print_transformed_nodes(nodes_trans, idx_trans_sorted)
        print(f"Sorted:\n{nodes_trans_sorted}")
        print(A_trans)

        np.testing.assert_equal(
            A_trans,
            np.array(
                [
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                ]
            ),
        )

    def test_both_flips(self):
        print("Both flips: vertical, then horizontal")

        print("\tVertical flip")

        nodes_trans1 = self._apply_vert_flip(self.nodes_orig)
        idx_sorted, nodes_trans1_sorted = sort_nodes(nodes_trans1)
        idx_trans1_sorted = np.take(self.idx_orig, idx_sorted)

        A_trans1 = self._transform_A(self.A, idx_trans1_sorted)
        self._print_transformed_nodes(nodes_trans1, idx_trans1_sorted)
        print(f"Sorted:\n{nodes_trans1_sorted}")

        print("\tHorizontal flip")

        nodes_trans2 = self._apply_horz_flip(nodes_trans1_sorted)
        idx_sorted, nodes_trans2_sorted = sort_nodes(nodes_trans2)
        idx_trans2_sorted = np.take(idx_trans1_sorted, idx_sorted)

        A_trans2 = self._transform_A(A_trans1, idx_trans2_sorted)
        self._print_transformed_nodes(nodes_trans2, idx_trans2_sorted)
        print(f"Sorted:\n{nodes_trans2_sorted}")
        print(A_trans2)

        np.testing.assert_equal(
            A_trans2,
            np.array(
                [
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                ]
            ),
        )

    def _apply_horz_flip(self, nodes: list) -> list:
        """Only columns/x-coords are flipped."""

        def flip_x(xy):
            x, y = xy
            return [self.max_x + self.min_x - x, y]

        return list(map(flip_x, nodes))

    def _apply_vert_flip(self, nodes: list) -> list:
        """Only rows/y-coords are flipped."""

        def flip_y(xy):
            x, y = xy
            return [x, self.max_y + self.min_y - y]

        return list(map(flip_y, nodes))
