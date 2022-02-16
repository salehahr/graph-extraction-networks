from typing import List, Tuple

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


def eedg_predict(eedg, model, step_num: int) -> Tuple:
    x, y_true = eedg[step_num]
    y_pred = model.predict(x)
    y_pred, _ = classify(y_pred)
    return y_true, y_pred


def eedg_coordinates(eedg, step_num: int) -> List[np.ndarray]:
    """Gets coordinates of the EEDG node combinations for the given step number."""
    pos_list = eedg.pos_list.numpy()
    combos = eedg.get_combo(step_num).numpy()

    return [pos_list[c] for c in combos]
