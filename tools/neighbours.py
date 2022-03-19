import tensorflow as tf

from tools.data import unique_2d


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
    ],
)
def distance(n1: tf.Tensor, n2: tf.Tensor) -> tf.Tensor:
    """Calculates the distance between two nodes."""
    diff = tf.cast(n1 - n2, tf.float32)
    return tf.math.reduce_euclidean_norm(diff, axis=1)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
    ]
)
def get_neighbours(
    num_neighbours, reduced_combos, pos_list_xy, nodes_not_found
) -> tf.Tensor:
    """Obtains k nearest neighbours from the available node combinations, without duplicates."""
    # make RaggedTensors to deal with cases when there are not enough neighbours
    combos = tf.map_fn(
        lambda x: get_nearest_neighbours(
            x, num_neighbours, reduced_combos, pos_list_xy
        ),
        elems=nodes_not_found,
        fn_output_signature=tf.RaggedTensorSpec(
            shape=[None, 2], ragged_rank=0, dtype=tf.int64
        ),
    )
    return unique_2d(combos.flat_values)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.int64),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
    ]
)
def get_nearest_neighbours(node_id, num_neighbours, all_combos, pos_list):
    """Returns the nearest neighbours of the node at node_id."""
    # length might be < num_neighbours if all_combos has become sparse (due to nodes being discarded)
    combos = get_node_neighbours(node_id, all_combos)
    combos_xy = tf.gather(pos_list, combos)

    # if num_combos < num_neighbours, no need to calculate nearest neighbours
    return tf.cond(
        tf.shape(combos_xy)[0] < num_neighbours,
        true_fn=lambda: combos,
        false_fn=lambda: nearest_neighbours(
            node_id, num_neighbours, all_combos, pos_list
        ),
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
    ],
)
def get_node_neighbours(node_id: tf.Tensor, combos: tf.Tensor):
    """Returns the node combinations that contain node_id."""
    idx_contains_node = tf.math.reduce_any(tf.equal(combos, node_id), axis=1)
    idx_contains_node = tf.where(idx_contains_node)
    num_vals = tf.shape(idx_contains_node)[0]

    combo_ids = tf.reshape(
        tf.gather(combos, tf.squeeze(idx_contains_node)), (num_vals, 2)
    )

    return combo_ids


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.int64),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
    ]
)
def nearest_neighbours(
    node_id: tf.Tensor,
    num_neighbours: tf.Tensor,
    all_combos: tf.Tensor,
    pos_list: tf.Tensor,
) -> tf.Tensor:
    """Returns the nearest neighbours of the node at node_id."""
    # length might be < num_neighbours if all_combos has become sparse (due to nodes being discarded)
    combos = get_node_neighbours(node_id, all_combos)
    combos_xy = tf.gather(pos_list, combos)

    # get the smallest distance
    combos_p1_xy, combos_p2_xy = combos_xy[:, 0, :], combos_xy[:, 1, :]
    distances = distance(combos_p1_xy, combos_p2_xy)
    _, indices = tf.math.top_k(-1 * distances, k=num_neighbours)

    return tf.gather(combos, indices)
