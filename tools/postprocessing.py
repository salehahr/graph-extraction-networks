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
