from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.lookup.experimental import DenseHashTable


def init_lookup_tables() -> Dict[str, DenseHashTable]:
    table_N_adjacency = DenseHashTable(
        key_dtype=tf.int64,
        value_dtype=tf.int64,
        default_value=-1,
        empty_key=-1,
        deleted_key=-2,
    )
    table_N_degrees = DenseHashTable(
        key_dtype=tf.int64,
        value_dtype=tf.int64,
        default_value=-1,
        empty_key=-1,
        deleted_key=-2,
    )
    table_N_found = DenseHashTable(
        key_dtype=tf.int64,
        value_dtype=tf.bool,
        default_value=False,
        empty_key=-1,
        deleted_key=-2,
    )

    # need this to determine unique combos when batching neighbour combinations
    table_NN_ids = DenseHashTable(
        key_dtype=tf.int64,
        value_dtype=tf.int64,
        default_value=-1,
        empty_key=[-1, -1],
        deleted_key=[-2, -2],
    )
    # table_NN_pairs = DenseHashTable(
    #     key_dtype=tf.int64,
    #     value_dtype=tf.int64,
    #     default_value=[-1, -1],
    #     empty_key=-1,
    #     deleted_key=-2,
    # )
    table_NN_adjacency = DenseHashTable(
        key_dtype=tf.int64,
        value_dtype=tf.int64,
        default_value=-1,
        empty_key=[-1, -1],
        deleted_key=[-2, -2],
    )
    table_NN_degrees = DenseHashTable(
        key_dtype=tf.int64,
        value_dtype=tf.int64,
        default_value=tf.constant([-1, -1], dtype=tf.int64),
        empty_key=tf.constant([-1, -1], dtype=tf.int64),
        deleted_key=tf.constant([-2, -2], dtype=tf.int64),
    )
    table_NN_found = DenseHashTable(
        key_dtype=tf.int64,
        value_dtype=tf.bool,
        default_value=[False, False],
        empty_key=[-1, -1],
        deleted_key=[-2, -2],
    )

    return {
        "N": {
            "adjacency": table_N_adjacency,
            "degrees": table_N_degrees,
            "found": table_N_found,
        },
        "NN": {
            "ids": table_NN_ids,
            # "pairs": table_NN_pairs,
            "adjacency": table_NN_adjacency,
            "degrees": table_NN_degrees,
            "found": table_NN_found,
        },
    }


def combo_lookup_tables(
    combos, adjacency_probs, adjacencies
) -> Tuple[DenseHashTable, DenseHashTable]:
    table_adjacency_probs = DenseHashTable(
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        default_value=-1.0,
        empty_key=[-1, -1],
        deleted_key=[-2, -2],
    )
    table_adjacencies = DenseHashTable(
        key_dtype=tf.int64,
        value_dtype=tf.int64,
        default_value=-1,
        empty_key=[-1, -1],
        deleted_key=[-2, -2],
    )

    table_adjacency_probs.insert(combos, adjacency_probs)
    table_adjacencies.insert(combos, adjacencies)

    return table_adjacency_probs, table_adjacencies
