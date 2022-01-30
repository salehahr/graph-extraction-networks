import numpy as np
import tensorflow as tf


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
