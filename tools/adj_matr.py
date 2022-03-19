from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from tools.plots import plot_adj_matr


def transform_A(A: np.ndarray, new_indices: np.ndarray) -> np.ndarray:
    """Generates new adjacency matrix based on new node positions."""
    num_nodes = A.shape[0]
    I = np.identity(num_nodes, dtype=np.uint8)
    P = np.take(I, new_indices, axis=0)
    return P @ A @ np.transpose(P)


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


@tf.function
def _update(combos: tf.Tensor, adjacencies: tf.Tensor, A: tf.Variable):
    A.scatter_nd_update(combos, adjacencies)
    A.scatter_nd_update(tf.reverse(combos, axis=[-1]), adjacencies)


def get_update_function(A: tf.Variable) -> tf.types.experimental.ConcreteFunction:
    return _update.get_concrete_function(
        combos=tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        adjacencies=tf.TensorSpec(shape=(None,), dtype=tf.int64),
        A=A,
    )


def get_placeholders(
    all_combos: tf.Tensor, degrees_list: tf.Tensor, num_nodes: tf.Tensor
) -> Tuple[tf.Variable, tf.Variable, tf.Variable]:
    adjacencies_var = tf.Variable(
        initial_value=tf.repeat(
            tf.constant(-1, dtype=tf.int64), repeats=tf.shape(all_combos)[0]
        ),
        trainable=False,
    )
    degrees_var = tf.Variable(initial_value=degrees_list, trainable=False)
    A = tf.Variable(
        initial_value=tf.zeros((num_nodes, num_nodes), dtype=tf.int64),
        trainable=False,
    )

    return adjacencies_var, degrees_var, A


def preview(
    adj_matr: tf.Variable,
    skel_img: tf.Tensor,
    pos_list_xy: tf.Tensor,
    title: Optional[str] = None,
):
    plot_adj_matr(
        skel_img.numpy(), pos_list_xy.numpy(), adj_matr.value().numpy(), title=title
    )
