import os
import unittest

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tools import PolyGraph
from tools.data import fp_to_adj_matr, fp_to_node_attributes
from tools.files import get_random_image, get_random_video_path
from tools.plots import plot_adj_matr, plot_bgr_img
from tools.sort import sort_list_of_nodes

data_path = os.path.join(os.getcwd(), "../data/test")

img_length = 256
base_path = f"/graphics/scratch/schuelej/sar/data/{img_length}"


class RandomImage(unittest.TestCase):
    """Class to get a random image from the data folder for testing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_path = base_path
        cls.video_path = get_random_video_path(base_path)

        img_name = get_random_image(cls.video_path)
        print(f"Video path: {cls.video_path}")
        print(f"Image: {img_name}")

        cls.is_synthetic = "synthetic" in cls.video_path
        cls.title = os.path.join(
            os.path.relpath(cls.video_path, start=base_path),
            os.path.splitext(img_name)[0],
        )

        cls.img_raw_fp = os.path.join(cls.video_path, f"raw/{img_name}")
        cls.img_skeletonised_fp = cls.img_raw_fp.replace("raw", "skeleton")

        cls.img_length = img_length

        assert os.path.isfile(cls.img_raw_fp)
        assert os.path.isfile(cls.img_skeletonised_fp)


class TestGraph(RandomImage):
    """
    Sanity checks:
        * tests whether the node positions are sorted
        * tests whether the adjacency matrix matches the skeletonised image
        * tests the classification of the nodes
    """

    @classmethod
    def setUpClass(cls) -> None:
        super(TestGraph, cls).setUpClass()

        cls.img_skel = cv2.imread(cls.img_skeletonised_fp, cv2.IMREAD_GRAYSCALE)
        plot_bgr_img(cls.img_skel, cls.title, show=True)

        graph_fp = cls.img_skeletonised_fp.replace("skeleton", "graphs").replace(
            ".png", ".json"
        )
        cls.graph = PolyGraph.load(graph_fp)

        cls.adj_matr = fp_to_adj_matr(graph_fp)

        node_pos = fp_to_node_attributes(graph_fp, img_length)[0, :, :, :].squeeze()
        positions = np.fliplr(np.argwhere(node_pos)).tolist()
        cls.positions = sort_list_of_nodes(positions)

    def plot_adj_matr(self, adj_matr):
        plot_adj_matr(self.img_skel, self.positions, adj_matr)
        plt.show()

    def test_is_pos_list_sorted(self):
        def is_sorted_ascending(arr):
            return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

        rows, cols = zip(*self.positions)
        print(self.positions)

        is_sorted_row = is_sorted_ascending(rows)

        self.assertTrue(is_sorted_row)

    def test_adjacency_matrix_skeletonised_match(self):
        if self.adj_matr is not None:
            self.plot_adj_matr(self.adj_matr)
