from typing import Union

import tensorflow as tf
from tensorflow.keras.losses import Loss, sparse_categorical_crossentropy


class WeightedCrossEntropy(Loss):
    """
    Implement class weights in the categorical cross entropy loss.
    https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
    """

    def __init__(self, n_classes: Union[int, tf.Tensor]):
        super(WeightedCrossEntropy, self).__init__()
        self._n_classes = n_classes

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # generate weights based on labels
        one_hot = tf.one_hot(y_true, depth=self._n_classes, dtype=tf.float64)
        weights = self._class_weights_in_batch(y_true)
        batch_weights = tf.reduce_sum(weights * one_hot, axis=-1)

        # adjust original loss function with batch_weights
        batch_losses = sparse_categorical_crossentropy(
            y_true, tf.cast(y_pred, tf.float64)
        )
        batch_losses = batch_losses * tf.squeeze(batch_weights)

        return batch_losses

    def _class_weights_in_batch(self, y_true: tf.Tensor) -> tf.Tensor:
        """
        For the labels of the current batch, generate weight values
        based on the frequency of class occurrence.
        """
        # y = tf.reshape(y_true, [-1])
        indices, _, count = tf.unique_with_counts(tf.reshape(y_true, [-1]))
        n_classes_in_batch = tf.shape(indices)[0]
        total = tf.reduce_sum(count)

        # each class weight is 1 / class_count * (total / n_classes)
        # multiplying by total / n_classes helps to normalise it
        weights = total / (count * n_classes_in_batch)

        # define class weights for batch
        # weights for classes not seen in current batch: init. to 1,
        #   later readjusted by normalisation over the whole weight vector.
        indices = tf.reshape(tf.cast(indices, tf.int64), (n_classes_in_batch, 1))
        batch_weights = tf.ones((self._n_classes,), dtype=tf.float64)
        batch_weights = tf.tensor_scatter_nd_add(batch_weights, indices, weights)

        # weight for 0-valued pixels might be too low
        # make it equal to the next minimum weight
        next_minimum = tf.expand_dims(tf.math.reduce_min(batch_weights[1:]), axis=-1)
        batch_weights = tf.tensor_scatter_nd_update(batch_weights, [[0]], next_minimum)

        # # normalisation (all weights sum up to one)
        # batch_weights = batch_weights / tf.reduce_sum(batch_weights)

        # make smallest weight 1
        batch_weights = batch_weights / tf.reduce_min(batch_weights)

        return batch_weights
