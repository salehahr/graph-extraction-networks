import unittest
from typing import List, Tuple

import networkx as nx
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from tools import Config, GraphExtractionDG, NodeExtractionDG, RunConfig, TestType
from tools.adj_matr import transform_A
from tools.plots import plot_augmented, plot_training_sample
from tools.sort import sort_nodes


class TestNodesNNDataAugmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config("test_config.yaml")
        cls.run_config = RunConfig("test_wandb_config.yaml", cls.config)

        cls.img_datagen = cls._init_old_data_augmenter()
        cls.aug_seed = 1

        cls.network = cls._set_network()
        cls.dg = cls._set_data_generator()
        cls.training_data = cls.dg(
            cls.config,
            cls.network,
            TestType.TRAINING,
            augmented=False,
        )

    @classmethod
    def _set_network(cls):
        return cls.config.network.node_extraction

    @classmethod
    def _set_data_generator(cls):
        return NodeExtractionDG

    @classmethod
    def _init_old_data_augmenter(cls) -> ImageDataGenerator:
        data_gen_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
            # rotation_range=90,
        )

        return ImageDataGenerator(**data_gen_args)

    def _get_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get first output of the first training batch."""
        x_batch, y_batch = self._get_first_batch()

        # expand dimensions for inputting to ImageDataGenerator.flow()
        x_first = np.expand_dims(x_batch[0].numpy(), axis=0)

        node_pos_first = np.expand_dims(y_batch[0][0].numpy(), axis=0)
        degrees_first = np.expand_dims(y_batch[1][0].numpy(), axis=0)
        node_types_first = np.expand_dims(y_batch[2][0].numpy(), axis=0)

        return x_first, node_pos_first, degrees_first, node_types_first

    def _get_first_batch(self) -> Tuple:
        """Gets first training batch."""
        plot_training_sample(
            self.training_data, network=self.network.id, rows=1, small_section=True
        )

        # gets the first batch (>= 1 images in batch)
        return self.training_data[0]

    @unittest.skip("Deprecated.")
    def test_old_data_aug(self):
        """
        For testing the method previously used for data augmentation.
        """
        samples = self._get_samples()

        # small section for report/graphics
        skel_img, node_pos, degrees, node_types = [
            s[:, 100:200, 100:200, :] for s in samples
        ]

        skel_img_aug = self._aug_imgs(skel_img)
        graph_aug = {
            "node_pos": self._aug_imgs(node_pos),
            "degrees": self._aug_imgs(degrees),
            "node_types": self._aug_imgs(node_types),
        }

        plot_augmented(skel_img_aug, graph_aug)

    def _aug_imgs(self, sample: np.ndarray) -> List[np.ndarray]:
        img_iter = self.img_datagen.flow(sample, batch_size=1, seed=self.aug_seed)
        return [img_iter.next() for _ in range(4)]

    def test_data_aug(self):
        plot_training_sample(self.training_data, network=self.network.id, rows=3)

        self.training_data.augmented = True

        plot_training_sample(self.training_data, network=self.network.id, rows=3)


class TestAdjMatrNNDataAugmentation(TestNodesNNDataAugmentation):
    @classmethod
    def _set_network(cls):
        return cls.config.network.graph_extraction

    @classmethod
    def _set_data_generator(cls):
        return GraphExtractionDG

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

    def _print_transformed_nodes(self, nodes, idx_new):
        for i, n, i_ts in zip(self.idx_orig, nodes, idx_new):
            print(f"{i}: {n} --> {i_ts}")

    def test_horz_flip(self):
        print("Horizontal flip")

        nodes_trans = self._apply_horz_flip(self.nodes_orig)
        idx_sorted, nodes_trans_sorted = sort_nodes(nodes_trans)
        idx_trans_sorted = np.take(self.idx_orig, idx_sorted)

        A_trans = transform_A(self.A, idx_trans_sorted)
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

        A_trans = transform_A(self.A, idx_trans_sorted)
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

        A_trans1 = transform_A(self.A, idx_trans1_sorted)
        self._print_transformed_nodes(nodes_trans1, idx_trans1_sorted)
        print(f"Sorted:\n{nodes_trans1_sorted}")

        print("\tHorizontal flip")

        nodes_trans2 = self._apply_horz_flip(nodes_trans1_sorted)
        idx_sorted, nodes_trans2_sorted = sort_nodes(nodes_trans2)
        idx_trans2_sorted = np.take(idx_trans1_sorted, idx_sorted)

        A_trans2 = transform_A(A_trans1, idx_trans2_sorted)
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

    def _apply_horz_flip(self, nodes: np.ndarray) -> list:
        """Only columns/x-coords are flipped."""

        def flip_x(xy):
            x, y = xy
            return [self.max_x + self.min_x - x, y]

        return list(map(flip_x, nodes))

    def _apply_vert_flip(self, nodes: np.ndarray) -> list:
        """Only rows/y-coords are flipped."""

        def flip_y(xy):
            x, y = xy
            return [x, self.max_y + self.min_y - y]

        return list(map(flip_y, nodes))
