from typing import Tuple

import tensorflow as tf

from tools.data import is_empty_tensor

EMPTY_TENSOR = tf.cast(tf.reshape((), (0)), tf.int64)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
    ],
)
def unique_nodes_from_combo(combos: tf.Tensor) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    """Returns a list of nodes present in the given node pair combinations without duplicates,
    as well as the corresponding indices relative to the list of node pair combinations."""
    # flatten combos
    num_elems_combos = tf.reduce_prod(
        tf.shape(combos, out_type=tf.int64), keepdims=True
    )
    nodes = tf.unique(tf.reshape(combos, num_elems_combos)).y

    # get row indices for each unique node
    rows_in_combos = tf.map_fn(
        lambda node: tf.where(combos == node)[:, 0],
        elems=nodes,
        fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int64),
    )

    return nodes, rows_in_combos


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ]
)
def combo_indices_without_nodes(combos: tf.Tensor, nodes: tf.Tensor) -> tf.Tensor:
    """Returns indices where the nodes are NOT found in combos."""
    # note: throws error if nodes is an empty Tensor
    not_found = tf.map_fn(
        lambda node: tf.not_equal(combos, node),
        elems=nodes,
        fn_output_signature=tf.bool,
    )

    return tf.where(tf.reduce_all(not_found, axis=[0, 2]))[:, 0]


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ]
)
def combos_without_nodes(
    nodes_to_remove: tf.Tensor, combos: tf.Tensor, adjacencies: tf.Tensor
):
    """Remove some nodes from the given set of node pair combinations."""
    not_found_indices = combo_indices_without_nodes(combos, nodes_to_remove)
    combos_new = tf.gather(combos, not_found_indices)
    adjacencies_new = tf.gather(adjacencies, not_found_indices)

    nodes, node_rows = unique_nodes_from_combo(combos_new)
    node_adjacencies = tf.reduce_sum(tf.gather(adjacencies_new, node_rows), axis=1)
    rows = tf.unique(node_rows.flat_values).y

    return combos_new, adjacencies_new, nodes, rows, node_adjacencies


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ]
)
def get_combo_nodes(
    combos: tf.Tensor,
    adjacencies: tf.Tensor,
    adjacency_probs: tf.Tensor,
    degrees_list: tf.Tensor,
):
    """Returns only the nodes (+ other node data) that are in the given node pair combinations."""
    nodes, node_rows = unique_nodes_from_combo(combos)
    node_adj, node_adj_probs = node_adjacencies(node_rows, adjacencies, adjacency_probs)
    node_degrees = tf.cast(tf.gather(degrees_list, nodes), tf.int64)

    return nodes, node_rows, node_adj, node_adj_probs, node_degrees


@tf.function(
    input_signature=[
        tf.RaggedTensorSpec(shape=None, dtype=tf.int64, ragged_rank=1),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ],
)
def node_adjacencies(
    rows_in_combos: tf.RaggedTensor, adjacencies: tf.Tensor, adjacency_probs: tf.Tensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    """Returns the corresponding adjacencies for nodes given their indices relative to
    a list of node pair combinations."""
    unique_adjacencies = tf.reduce_sum(tf.gather(adjacencies, rows_in_combos), axis=1)
    unique_adj_probs = tf.gather(adjacency_probs, rows_in_combos)

    return unique_adjacencies, unique_adj_probs


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
    ]
)
def remove_combo_subset(
    combos: tf.Tensor,
    adjacencies: tf.Tensor,
    adjacency_probs: tf.Tensor,
    subset: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Given combos of dimension [c, 2] and a subset of it with dimensions [s, 2],
    returns the complement of the intersection between the two.
    """

    # this reduction acts on the pair_id axis, resulting in a [s, c] boolean matrix.
    bool_mask = tf.map_fn(
        lambda x: tf.reduce_all(tf.equal(combos, x), axis=-1),
        elems=subset,
        fn_output_signature=tf.bool,
    )

    # this reduction acts across the <c> values crossed with each <s> value
    indices = tf.where(tf.reduce_all(tf.logical_not(bool_mask), axis=0))[:, 0]

    return (
        tf.gather(combos, indices),
        tf.gather(adjacencies, indices),
        tf.gather(adjacency_probs, indices),
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
    ]
)
def remove_combo_subset_from_all(
    combos: tf.Tensor,
    subset: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Given combos of dimension [c, 2] and a subset of it with dimensions [s, 2],
    returns the complement of the intersection between the two.
    """

    # this reduction acts on the pair_id axis, resulting in a [s, c] boolean matrix.
    bool_mask = tf.map_fn(
        lambda x: tf.reduce_all(tf.equal(combos, x), axis=-1),
        elems=subset,
        fn_output_signature=tf.bool,
    )

    # this reduction acts across the <c> values crossed with each <s> value
    indices = tf.where(tf.reduce_all(tf.logical_not(bool_mask), axis=0))[:, 0]

    return tf.gather(combos, indices)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ]
)
def remove_nodes_found(
    all_combos: tf.Tensor,
    combos: tf.Tensor,
    adjacencies: tf.Tensor,
    adjacency_probs: tf.Tensor,
    nodes_found: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    all_not_found_indices = combo_indices_without_nodes(all_combos, nodes_found)
    not_found_indices = combo_indices_without_nodes(combos, nodes_found)
    return (
        tf.gather(all_combos, all_not_found_indices),
        tf.gather(combos, not_found_indices),
        tf.gather(adjacencies, not_found_indices),
        tf.gather(adjacency_probs, not_found_indices),
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ]
)
def remove_conflicting(
    combos: tf.Tensor, adjacencies: tf.Tensor, degrees_list: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Removes node combinations with conflicts."""

    # all nodes in combos
    nodes, node_rows = unique_nodes_from_combo(combos)
    rows = tf.unique(node_rows.flat_values).y
    node_adjacencies = tf.reduce_sum(tf.gather(adjacencies, node_rows), axis=1)

    # degrees check
    # new_degrees might be < 0!
    # e.g. for combinations [a, b], [a, c] with predicted adjacencies 1 each:
    # if, currently, deg(a) = 1, deg(b) = 1 and deg(c) = 1
    # then new_deg(a) = -1 if the conflict is not resolved
    node_degrees = tf.gather(degrees_list, nodes)
    new_degrees = node_degrees - node_adjacencies
    bad_nodes = tf.gather(nodes, tf.where(new_degrees < 0)[:, 0])
    bad_nodes_exist = tf.size(bad_nodes) != 0

    # trim combos
    combos_, adjacencies_, nodes_, rows_, node_adjacencies_ = tf.cond(
        bad_nodes_exist,
        true_fn=lambda: combos_without_nodes(bad_nodes, combos, adjacencies),
        false_fn=lambda: (combos, adjacencies, nodes, rows, node_adjacencies),
    )

    return combos_, adjacencies_, nodes_, rows_, node_adjacencies_


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.int64)])
def nodes_found(degrees: tf.Tensor) -> tf.Tensor:
    """Returns nodes where degrees are already zero."""
    return tf.squeeze(tf.where(tf.equal(degrees, 0)))


def update_lookup(
    adjacencies_var, degrees_var, combo_ids, adjacencies, nodes, node_adjacencies
):
    # update adjacencies
    adjacencies_var.scatter_nd_update(tf.expand_dims(combo_ids, axis=-1), adjacencies)

    # update degrees
    node_degrees = tf.gather(degrees_var, nodes)
    new_degrees = node_degrees - node_adjacencies
    degrees_var.scatter_nd_update(tf.expand_dims(nodes, axis=-1), new_degrees)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.RaggedTensorSpec(shape=None, dtype=tf.int64, ragged_rank=1),
        tf.RaggedTensorSpec(shape=None, dtype=tf.float32, ragged_rank=1),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ]
)
def get_capped_combos(combos, node_rows, node_adj_probs, node_degrees):
    # cap adjacencies based on degree
    node_adjacencies_capped = get_new_adjacencies(node_adj_probs, node_degrees)

    # select which combos to proceed with (discard if duplicate values are problematic)
    return get_combos_to_keep(combos, node_rows, node_adjacencies_capped)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.RaggedTensorSpec(shape=None, dtype=tf.int64, ragged_rank=1),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.RaggedTensorSpec(shape=None, dtype=tf.float32, ragged_rank=1),
    ],
)
def filter_nodes_by_case(
    comparison_adj_degrees: tf.bool,
    nodes: tf.Tensor,
    rows: tf.RaggedTensor,
    adjacencies: tf.Tensor,
    adjacency_probs: tf.RaggedTensor,
) -> Tuple[tf.Tensor, tf.RaggedTensor, tf.Tensor, tf.RaggedTensor]:
    # indices relative to unique_nodes or unique_rows
    indices = tf.where(comparison_adj_degrees)[:, 0]

    # ragged indices:
    # first axis is relative to nodes, second axis contains indices relative to combos
    nodes = tf.gather(nodes, indices)
    rows = tf.gather(rows, indices)
    adjacencies = tf.gather(adjacencies, indices)
    adjacency_probs = tf.gather(adjacency_probs, indices)

    return nodes, rows, adjacencies, adjacency_probs


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.RaggedTensorSpec(shape=None, dtype=tf.int64, ragged_rank=1),
    ]
)
def combos_from_node_rows(combos, adjacencies, rows_in_combos):
    rows = tf.unique(rows_in_combos.flat_values).y
    combos_ = tf.gather(combos, rows)
    adjacencies_ = tf.gather(adjacencies, rows)

    return combos_, adjacencies_


def _predict_ok_good(
    comparison,
    nodes,
    node_rows,
    node_adj,
    node_adj_probs,
    combos,
    adjacencies,
    adjacencies_var,
    degrees_var,
):
    # only get nodes which match current comparison type
    nodes, node_rows, _, _ = filter_nodes_by_case(
        comparison, nodes, node_rows, node_adj, node_adj_probs
    )

    # get corresponding combinations
    case_combos, case_adj = combos_from_node_rows(combos, adjacencies, node_rows)

    # remove conclicting combinations
    (
        case_combos_,
        case_adj_,
        nodes_,
        rows_,
        node_adjacencies_,
    ) = remove_conflicting(case_combos, case_adj, degrees_var.value())

    # update lookup
    update_lookup(
        adjacencies_var, degrees_var, rows_, case_adj_, nodes_, node_adjacencies_
    )

    return case_combos_, case_adj_


def _predict_bad(
    comparison,
    nodes,
    node_rows,
    node_adj,
    node_adj_probs,
    combos,
    adjacencies_var,
    degrees_var,
):
    # only get nodes which match current comparison type
    nodes, node_rows, _, node_adj_probs = filter_nodes_by_case(
        comparison, nodes, node_rows, node_adj, node_adj_probs
    )
    node_degrees = tf.gather(degrees_var, nodes)

    # get corresponding combinations
    # remove combos based on new capped adjacencies
    case_combos, case_adj, _ = get_capped_combos(
        combos, node_rows, node_adj_probs, node_degrees
    )

    # remove conclicting combinations
    (
        case_combos_,
        case_adj_,
        nodes_,
        rows_,
        node_adjacencies_,
    ) = remove_conflicting(case_combos, case_adj, degrees_var.value())

    # update lookup
    update_lookup(
        adjacencies_var, degrees_var, rows_, case_adj_, nodes_, node_adjacencies_
    )

    return case_combos_, case_adj_


def predict_ok_good(
    nodes,
    node_rows,
    node_adj,
    node_adj_probs,
    node_degrees,
    combos,
    adjacencies,
    adjacencies_var,
    degrees_var,
):
    comparison = node_adj <= node_degrees
    exist_nodes = tf.reduce_any(comparison)
    case_combos, case_adj = tf.cond(
        exist_nodes,
        true_fn=lambda: _predict_ok_good(
            comparison,
            nodes,
            node_rows,
            node_adj,
            node_adj_probs,
            combos,
            adjacencies,
            adjacencies_var,
            degrees_var,
        ),
        false_fn=lambda: (EMPTY_TENSOR, EMPTY_TENSOR),
    )
    return case_combos, case_adj


def predict_bad(
    nodes,
    node_rows,
    node_adj,
    node_adj_probs,
    node_degrees,
    combos,
    adjacencies_var,
    degrees_var,
):
    comparison = node_adj > node_degrees
    exist_nodes = tf.reduce_any(comparison)
    case_combos, case_adj = tf.cond(
        exist_nodes,
        true_fn=lambda: _predict_bad(
            comparison,
            nodes,
            node_rows,
            node_adj,
            node_adj_probs,
            combos,
            adjacencies_var,
            degrees_var,
        ),
        false_fn=lambda: (EMPTY_TENSOR, EMPTY_TENSOR),
    )
    return case_combos, case_adj


@tf.function(
    input_signature=[
        tf.RaggedTensorSpec(shape=None, dtype=tf.float32, ragged_rank=1),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ],
)
def get_new_adjacencies(
    node_adj_probs: tf.RaggedTensor, node_degrees: tf.Tensor
) -> tf.RaggedTensor:
    """Returns new adjacency vector with <degree> of the highest probability entries."""
    adjacencies = tf.map_fn(
        new_adjacencies,
        elems=(node_adj_probs, node_degrees),
        fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32),
    )
    return tf.cast(adjacencies, tf.int64)


def new_adjacencies(args: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Returns new adjacency vector with <degree> of the highest probability entries."""
    probs, degree = args[0], args[1]
    max_prob_indices = tf.nn.top_k(probs, k=tf.cast(degree, tf.int32)).indices

    adj = tf.ones(tf.shape(max_prob_indices), dtype=tf.int32)
    shape = tf.shape(probs)

    new_adj = tf.scatter_nd(tf.expand_dims(max_prob_indices, axis=-1), adj, shape)
    return new_adj


def get_combos_to_keep(
    combos: tf.Tensor, node_rows: tf.RaggedTensor, node_adjacencies: tf.RaggedTensor
):
    non_duplicate_combo_ids, duplicate_combo_ids = check_duplicate_combo_ids(node_rows)
    # unique_combo_ids, _, counts = tf.unique_with_counts(node_rows.flat_values)
    # non_duplicate_combo_ids = tf.gather(unique_combo_ids, tf.where(counts == 1))[:, 0]
    # duplicate_combo_ids = tf.gather(unique_combo_ids, tf.where(counts > 1))[:, 0]

    # for checking if empty
    exist_non_duplicates = tf.logical_not(is_empty_tensor(non_duplicate_combo_ids))
    exist_duplicates = tf.logical_not(is_empty_tensor(duplicate_combo_ids))

    # indices relative to node_rows.flat_values
    # [n_combos, 1]
    _non_dup_flat_indices = get_indices_of_combos_in_node_rows(
        node_rows, non_duplicate_combo_ids
    )
    # [n_combos, n_dups (normally 2) , 1]
    _dup_flat_indices = get_indices_of_combos_in_node_rows(
        node_rows, duplicate_combo_ids
    )

    # adjacencies
    if exist_non_duplicates:
        non_dup_adjs = tf.squeeze(
            tf.gather(node_adjacencies.flat_values, _non_dup_flat_indices)
        )
    else:
        non_dup_adjs = EMPTY_TENSOR

    if exist_duplicates:
        dup_adjs = tf.gather(node_adjacencies.flat_values, _dup_flat_indices)[..., 0]
        dup_is_same_adj = tf.map_fn(
            lambda x: tf.reduce_all(tf.equal(tf.reduce_mean(x), x)),
            elems=dup_adjs,
            fn_output_signature=tf.bool,
        )

        # indices relative to duplicate_combo_ids
        _dups_to_discard_idcs = tf.where(dup_is_same_adj == False)[:, 0]
        _dups_to_keep_idcs = tf.where(dup_is_same_adj)[:, 0]

        combo_ids_to_discard = tf.gather(duplicate_combo_ids, _dups_to_discard_idcs)
        exist_nodes_to_discard = tf.not_equal(tf.shape(combo_ids_to_discard), 0)
        # if not exist_nodes_to_discard:
        #     return combos, tf.gather(node_adjacencies.flat_values, )

        # take non-discarded duplicate values
        combo_ids = tf.gather(duplicate_combo_ids, _dups_to_keep_idcs)
        adjacencies = tf.gather(dup_adjs[:, 0], _dups_to_keep_idcs)
    else:
        dup_adjs = EMPTY_TENSOR
        _dups_to_keep_idcs = EMPTY_TENSOR
        combo_ids = EMPTY_TENSOR
        adjacencies = EMPTY_TENSOR
        exist_nodes_to_discard = False

    # take non-discarded duplicate values
    combo_ids = tf.gather(duplicate_combo_ids, _dups_to_keep_idcs)
    adjacencies = (
        EMPTY_TENSOR
        if is_empty_tensor(dup_adjs)
        else tf.gather(dup_adjs[:, 0], _dups_to_keep_idcs)
    )

    # concat with non-duplicates
    combo_ids = tf.concat((non_duplicate_combo_ids, combo_ids), axis=0)
    adjacencies = tf.concat((non_dup_adjs, adjacencies), axis=0)

    combos_to_keep = tf.gather(combos, combo_ids)

    # discard
    if exist_nodes_to_discard:
        node_row_idcs_to_discard = tf.gather(_dup_flat_indices, _dups_to_discard_idcs)

        discard_shape = (
            tf.reduce_prod(tf.shape(node_row_idcs_to_discard, out_type=tf.int64)),
            1,
        )

        adj_diff = tf.cast(
            tf.gather(node_adjacencies.flat_values, node_row_idcs_to_discard),
            tf.int64,
        )
        adj_diff = tf.reshape(adj_diff, discard_shape)[:, 0]

        node_row_idcs_to_discard = tf.reshape(node_row_idcs_to_discard, discard_shape)
        adj_flat_shape = tf.shape(node_adjacencies.flat_values, out_type=tf.int64)

        adj_diff = node_adjacencies.flat_values - tf.scatter_nd(
            node_row_idcs_to_discard, adj_diff, adj_flat_shape
        )
        node_adjacencies_new = tf.RaggedTensor.from_row_splits(
            adj_diff, node_adjacencies.row_splits
        )
    else:
        node_adjacencies_new = node_adjacencies

    node_adjacencies_sum = tf.reduce_sum(node_adjacencies_new, axis=1)

    return combos_to_keep, tf.cast(adjacencies, tf.int64), node_adjacencies_sum


@tf.function(
    input_signature=[tf.RaggedTensorSpec(shape=None, dtype=tf.int64, ragged_rank=1)],
)
def check_duplicate_combo_ids(node_rows: tf.RaggedTensor):
    unique_combo_ids, _, counts = tf.unique_with_counts(node_rows.flat_values)
    non_duplicates_ids = tf.gather(unique_combo_ids, tf.where(counts == 1))[:, 0]
    duplicates_ids = tf.gather(unique_combo_ids, tf.where(counts > 1))[:, 0]

    return non_duplicates_ids, duplicates_ids


@tf.function(
    input_signature=[
        tf.RaggedTensorSpec(shape=None, dtype=tf.int64, ragged_rank=1),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ],
)
def get_indices_of_combos_in_node_rows(
    node_rows: tf.RaggedTensor, combo_ids: tf.Tensor
) -> tf.Tensor:
    """Returns indices of flattened tensor"""
    if is_empty_tensor(combo_ids):
        return combo_ids
    else:
        return tf.map_fn(
            lambda x: tf.where(node_rows.flat_values == x),
            elems=combo_ids,
        )


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.int64)])
def nodes_not_found(degrees: tf.Tensor) -> tf.Tensor:
    return tf.where(tf.not_equal(degrees, 0))
