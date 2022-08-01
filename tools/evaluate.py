from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import tensorflow as tf

from tools.DataGenerator import get_eedg, get_nedg
from tools.NetworkType import NetworkType
from tools.postprocessing import tf_classify

if TYPE_CHECKING:
    from model import EdgeNN
    from tools.config import Config, RunConfig
    from tools.DataGenerator import EdgeDG, NodeExtractionDG


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

    precision_pos, recall_pos = binary_metrics(pos_true, pos_pred)
    precision_degs, recall_degs = multilass_metrics(degs_true, degs_pred)
    precision_types, recall_types = multilass_metrics(types_true, types_pred)

    return int(n_nodes), {
        "loss": float(L),
        "L_pos": float(L_pos),
        "L_degs": float(L_degs),
        "L_types": float(L_types),
        "acc_pos": float(acc_pos),
        "precision_pos": float(precision_pos),
        "recall_pos": float(recall_pos),
        "acc_degs": float(acc_degs),
        "precision_degs": float(precision_degs),
        "recall_degs": float(recall_degs),
        "acc_types": float(acc_types),
        "precision_types": float(precision_types),
        "recall_types": float(recall_types),
    }


def binary_metrics(pos_true: tf.Tensor, pos_pred: tf.Tensor) -> Tuple[tf.Tensor, ...]:
    true = tf.squeeze(pos_true)
    pred = tf.squeeze(pos_pred)

    # tp = tf.keras.metrics.TruePositives()(true, pred)
    # fp = tf.keras.metrics.FalsePositives()(true, pred)
    # tn = tf.keras.metrics.TrueNegatives()(true, pred)
    # fn = tf.keras.metrics.FalseNegatives()(true, pred)

    precision = tf.keras.metrics.Precision()(true, pred)
    recall = tf.keras.metrics.Recall()(true, pred)

    return precision, recall


def multilass_metrics(
    true_tensor: tf.Tensor, pred_tensor: tf.Tensor
) -> Tuple[tf.Tensor, ...]:
    true = tf.reshape(tf.squeeze(true_tensor), -1)
    pred = tf.math.argmax(tf.squeeze(pred_tensor), axis=-1)
    pred = tf.reshape(pred, -1)

    confmatr = tf.math.confusion_matrix(true, pred)
    num_classes = tf.shape(confmatr)[0]

    precision = tf.constant(0, dtype=tf.float64)
    recall = tf.constant(0, dtype=tf.float64)
    for i in range(num_classes):
        row = confmatr[i, :]
        col = confmatr[:, i]

        if tf.equal(tf.reduce_sum(row), 0):
            recall += 1
        else:
            recall += row[i] / tf.reduce_sum(row)

        if tf.equal(tf.reduce_sum(col), 0):
            precision += 1
        else:
            precision += row[i] / tf.reduce_sum(col)

    num_classes = tf.cast(num_classes, tf.float64)
    macro_recall = recall / num_classes
    macro_precision = precision / num_classes
    # macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    return macro_recall, macro_precision  # , macro_f1


def get_test_data(
    network: NetworkType, data_config: Config, eval_config: RunConfig
) -> Tuple[Union[EdgeDG, NodeExtractionDG], List[str]]:
    if network == NetworkType.EDGE_NN:
        test_data = get_eedg(data_config, eval_config, test=True)
        metric_headers = ["tp", "tn", "fp", "fn", "precision", "recall", "f1"]
    elif network == NetworkType.NODES_NN:
        test_data = get_nedg(data_config, test=True)
        metric_headers = [
            "loss",
            "L_pos",
            "L_degs",
            "L_types",
            "acc_pos",
            "precision_pos",
            "recall_pos",
            "acc_degs",
            "precision_degs",
            "recall_degs",
            "acc_types",
            "precision_types",
            "recall_types",
        ]
    return test_data, metric_headers
