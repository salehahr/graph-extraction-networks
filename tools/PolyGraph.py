import json

import networkx as nx
import numpy as np


class PolyGraph(nx.Graph):
    """
    Graph with polynomial edge attributes.
    (not a lie detector)
    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    @classmethod
    def load(cls, filepath: str) -> nx.Graph:
        """Loads a graph from filepath."""
        with open(filepath, "r") as f:
            data_dict = json.load(f)
            graph = nx.node_link_graph(data_dict)
        return cls(graph)

    # properties for node extraction NN
    @property
    def positions(self) -> list:
        return list(nx.get_node_attributes(self, "pos").values())

    @property
    def num_node_neighbours(self) -> list:
        return [v for _, v in self.degree]

    @property
    def node_types(self) -> list:
        orig_list = list(nx.get_node_attributes(self, "type").values())
        # change zeros to threes
        return [3 if i == 0 else i for i in orig_list]

    # properties for edge extraction NN
    @property
    def positions_vector(self) -> np.ndarray:
        return np.array(self.positions)

    @property
    def adj_matrix(self) -> np.ndarray:
        return nx.convert_matrix.to_numpy_array(self)

    @property
    def length_matrix(self) -> np.ndarray:
        return np.hstack(
            np.zeros(self.positions_vector.shape),
            nx.attr_matrix(self, edge_attr="length")[0],
        )

    @property
    def coeff3_matrix(self) -> np.ndarray:
        return np.hstack(
            np.zeros(self.positions_vector.shape),
            nx.attr_matrix(self, edge_attr="deg3")[0],
        )

    @property
    def coeff2_matrix(self) -> np.ndarray:
        return np.hstack(
            np.zeros(self.positions_vector.shape),
            nx.attr_matrix(self, edge_attr="deg3")[0],
        )

    def stacked_adj_matrix(self) -> np.ndarray:
        num_nodes = len(self)
        stacked_matrix = np.zeros((4, num_nodes, 2 + num_nodes))

        stacked_matrix[0, :2, :] = self.positions_vector

        stacked_matrix[0, 2:, :] = self.adj_matrix
        stacked_matrix[1, :, :] = self.length_matrix
        stacked_matrix[2, :, :] = self.coeff3_matrix
        stacked_matrix[3, :, :] = self.coeff2_matrix

        return stacked_matrix
