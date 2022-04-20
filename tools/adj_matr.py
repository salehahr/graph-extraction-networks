from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import tensorflow as tf

from tools.logger import Logger
from tools.NetworkType import NetworkType
from tools.plots import plot_adj_matr

if TYPE_CHECKING:
    from tools import AdjMatrPredictor, GraphExtractionDG


# noinspection PyPep8Naming
def transform_A(A: np.ndarray, new_indices: np.ndarray) -> np.ndarray:
    """Generates new adjacency matrix based on new node positions."""
    num_nodes = A.shape[0]
    I = np.identity(num_nodes, dtype=np.uint8)
    P = np.take(I, new_indices, axis=0)
    return P @ A @ np.transpose(P)


# noinspection PyPep8Naming
def transform_adj_matr(A: tf.Tensor, new_indices: tf.Tensor) -> tf.RaggedTensor:
    """Generates new adjacency matrix based on new node positions."""
    I = tf.eye(tf.shape(A)[0], dtype=A.dtype)
    P = tf.gather(I, tf.cast(new_indices, A.dtype))
    A_new = P @ tf.squeeze(A) @ tf.transpose(P)
    return tf.RaggedTensor.from_tensor(A_new)


def adj_matr_to_vec(adj_matr: np.ndarray) -> np.ndarray:
    upper_tri_matr = np.triu(adj_matr)
    upper_tri_idxs = np.triu_indices(adj_matr.shape[0], k=1)
    return upper_tri_matr[upper_tri_idxs]


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
def tf_A_vec(adj_matr: tf.Tensor) -> tf.Tensor:
    return tf.numpy_function(adj_matr_to_vec, inp=[adj_matr], Tout=tf.int32)


# noinspection PyPep8Naming
@tf.function
def _update(combos: tf.Tensor, adjacencies: tf.Tensor, A: tf.Variable):
    A.scatter_nd_update(combos, adjacencies)
    A.scatter_nd_update(tf.reverse(combos, axis=[-1]), adjacencies)


# noinspection PyPep8Naming
def get_update_function(A: tf.Variable) -> tf.types.experimental.ConcreteFunction:
    return _update.get_concrete_function(
        combos=tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        adjacencies=tf.TensorSpec(shape=(None,), dtype=tf.int64),
        A=A,
    )


def get_placeholders(
    degrees_list: tf.Tensor, num_nodes: tf.Tensor
) -> Tuple[tf.Variable, tf.Variable]:
    degrees_var = tf.Variable(initial_value=degrees_list, trainable=False)
    A = tf.Variable(
        initial_value=tf.zeros((num_nodes, num_nodes), dtype=tf.int64),
        trainable=False,
    )

    return degrees_var, A


def preview(
    adj_matr: tf.Variable,
    skel_img: tf.Tensor,
    pos_list_xy: tf.Tensor,
    title: Optional[str] = None,
):
    plot_adj_matr(
        skel_img.numpy(), pos_list_xy.numpy(), adj_matr.value().numpy(), title=title
    )


def leerlauf(predictor: AdjMatrPredictor, graph_data: GraphExtractionDG):
    # leerlauf
    print("Leerlauf.")
    edge_nn_input, _, _ = graph_data.get_single_data_point(0)
    predictor.predict(edge_nn_input)


def plot_in_loop(predictor: AdjMatrPredictor, graph_data: GraphExtractionDG):
    leerlauf(predictor, graph_data)

    # only save the matrices/skel_imgs, plot later
    print("Start of loop.")
    plot_array = []
    for i in range(0, 6):
        edge_nn_inputs, _, _ = graph_data.get_single_data_point(i)
        plot_data = predictor.predict(edge_nn_inputs)
        plot_array.append(plot_data)

    for pd in plot_array:
        preview(*pd)


def predict_loop(predictor: AdjMatrPredictor, graph_data: GraphExtractionDG):
    leerlauf(predictor, graph_data)

    metric_headers = ["tp", "tn", "fp", "fn", "precision", "recall", "f1"]
    logger = Logger(
        f"adj_pred-k{predictor.k0}.csv",
        headers=metric_headers,
        network=NetworkType.ADJ_MATR_NN,
    )

    print("Start of loop.")
    for i in range(len(graph_data)):
        combo_img, A_true, img = graph_data.get_single_data_point(i)
        A_pred, _, _, time = predictor.predict(combo_img, with_time=True)

        metrics = predictor.metrics(A_true[0].to_tensor())
        logger.write(metrics, img_fp=img, num_nodes=predictor.num_nodes, time=time)


def predict_first_batch(predictor: AdjMatrPredictor, graph_data: GraphExtractionDG):
    combo_img, A_true, img = graph_data.get_single_data_point(4)

    for i in range(5):
        predictor._init_prediction(*combo_img)

    predictor._adjacency_probs, predictor._adjacencies
