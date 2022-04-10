from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

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
