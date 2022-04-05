from typing import Tuple

import numpy as np
import tensorflow as tf


def classify(mask: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Returns mask with integer classes."""
    is_binary = mask.shape[-1] <= 2

    if is_binary:
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = mask.astype(np.uint8)
    else:
        mask = tf.argmax(mask, axis=-1)
        mask = mask[..., tf.newaxis].numpy()

    return mask, is_binary


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32)])
def tf_classify(output: tf.Tensor) -> tf.Tensor:
    return tf.cast(tf.greater(output, tf.constant(0.5)), tf.int64)


def smooth(vals: list, k: int = 10) -> np.ndarray:
    arr_plc = np.zeros((len(vals),))
    arr = np.array(vals)

    for i, a in enumerate(arr):
        if i < k:
            arr_plc[i] = np.mean(arr[0:k])
        else:
            arr_plc[i] = np.mean(arr[i - k + 1 : i + 1])

    return arr_plc
