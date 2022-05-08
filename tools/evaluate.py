from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import tensorflow as tf

from tools.postprocessing import tf_classify

if TYPE_CHECKING:
    from model import EdgeNN


def get_edgenn_caller(model: EdgeNN) -> tf.types.experimental.ConcreteFunction:
    def evaluate(
        model: EdgeNN, skel_img: tf.Tensor, node_pos: tf.Tensor, combo_img: tf.Tensor
    ):
        return model((skel_img, node_pos, combo_img), training=False)

    return tf.function(evaluate).get_concrete_function(
        model=model,
        skel_img=tf.TensorSpec(shape=(None, 256, 256), dtype=tf.float32),
        node_pos=tf.TensorSpec(shape=(None, 256, 256), dtype=tf.uint8),
        combo_img=tf.TensorSpec(shape=(None, 256, 256), dtype=tf.int64),
    )


@tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.float32)])
def classify(probs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    adjacency_probs = tf.squeeze(probs)
    adjacencies = tf_classify(adjacency_probs)
    return adjacency_probs, adjacencies


def nodes_nn_metrics(
    y_true: tf.Tensor, y_pred: tf.Tensor
) -> Tuple[int, Dict[str, float]]:
    pos_pred, degs_pred, types_pred = y_pred
    pos_true, degs_true, types_true = y_true
    n_nodes = tf.math.count_nonzero(pos_true)

    L_pos = tf.keras.metrics.BinaryCrossentropy()(pos_true, pos_pred)
    L_degs = tf.keras.metrics.SparseCategoricalCrossentropy()(degs_true, degs_pred)
    L_types = tf.keras.metrics.SparseCategoricalCrossentropy()(types_true, types_pred)
    L = L_pos + L_degs + L_types

    acc_pos = tf.keras.metrics.BinaryAccuracy()(pos_true, pos_pred)
    acc_degs = tf.keras.metrics.SparseCategoricalAccuracy()(degs_true, degs_pred)
    acc_types = tf.keras.metrics.SparseCategoricalAccuracy()(types_true, types_pred)

    return int(n_nodes), {
        "loss": float(L),
        "L_pos": float(L_pos),
        "L_degs": float(L_degs),
        "L_types": float(L_types),
        "acc_pos": float(acc_pos),
        "acc_degs": float(acc_degs),
        "acc_types": float(acc_types),
    }
