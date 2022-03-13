import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from tools.PolyGraph import PolyGraph
from tools.sort import get_sort_indices, sort_list_of_nodes


def get_skeletonised_ds(
    data_path: str, seed: int, is_test: bool = False
) -> tf.data.Dataset:
    """
    Returns skeletonised image paths.
    :param data_path: folder where the data is stored
    :param seed: seed for shuffling function, set None for random
    :param is_test: True if test set filepaths are to be used
    :return: the skeletonised image paths as a Dataset object
    """
    if is_test:
        skeletonised_files_glob = [
            os.path.join(data_path, "test_*/skeleton/*.png"),
            os.path.join(data_path, "*/test_*/skeleton/*.png"),
        ]
    else:
        skeletonised_files_glob = [
            os.path.join(data_path, "[!t]*/skeleton/*.png"),
            os.path.join(data_path, "*/[!t]*/skeleton/*.png"),
        ]

    ds = tf.data.Dataset.list_files(skeletonised_files_glob, shuffle=False)

    return ds.shuffle(len(ds), seed=seed, reshuffle_each_iteration=False)


def ds_to_list(dataset: tf.data.Dataset) -> List[str]:
    return [f.decode("utf-8") for f in dataset.as_numpy_iterator()]


@tf.function
def fp_to_grayscale_img(fp: tf.Tensor) -> tf.Tensor:
    raw_img = tf.io.read_file(fp)
    unscaled_img = tf.image.decode_png(raw_img, channels=1, dtype=tf.uint8)
    return tf.image.convert_image_dtype(unscaled_img, tf.float32)


def fp_to_node_attributes(fp: str, dim: int) -> np.ndarray:
    graph = PolyGraph.load(fp)
    return graph_to_node_attributes(graph, dim)


def graph_to_node_attributes(graph: PolyGraph, dim: int) -> np.ndarray:
    """
    Generates output matrices of the graph's node attributes.
    """
    node_attributes = np.zeros((3, dim, dim, 1)).astype(np.uint8)

    node_pos = node_attributes[0, :, :, :]
    degrees = node_attributes[1, :, :, :]
    node_types = node_attributes[2, :, :, :]

    for i, (col, row) in enumerate(graph.positions):
        node_pos[row][col] = 1
        degrees[row][col] = graph.num_node_neighbours[i]
        node_types[row][col] = graph.node_types[i]

    def cap_degrees(deg_matrix: np.ndarray) -> np.ndarray:
        """Cap values at 4."""
        cap_value = 4
        deg_matrix[deg_matrix > cap_value] = cap_value
        return deg_matrix

    degrees[:, :, :] = cap_degrees(degrees)

    return node_attributes


def fp_to_adj_matr(fp: str) -> np.ndarray:
    adj_matr = PolyGraph.load(fp).adj_matrix.astype(np.uint8)
    adj_matr = np.expand_dims(adj_matr, -1)
    return adj_matr


def tf_fp_to_adj_matr(fp: str):
    return tf.numpy_function(func=fp_to_adj_matr, inp=[fp], Tout=tf.uint8)


def pos_list_from_image(node_pos_img: np.ndarray) -> np.ndarray:
    """Gets list of coordinates from the node_pos image."""
    # flip to convert (row, col) to (x, y)
    pos_list_xy = np.fliplr(np.argwhere(node_pos_img)).tolist()
    return sort_list_of_nodes(pos_list_xy)


@tf.function
def sorted_pos_list_from_image(node_pos_img: tf.Tensor) -> tf.Tensor:
    """Extracts the sorted xy coordinates from the node_pos image."""
    xy_unsorted = unsorted_pos_list_from_image(node_pos_img)
    sort_indices = get_sort_indices(xy_unsorted)
    return tf.gather(xy_unsorted, sort_indices)


@tf.function
def indices_for_pos_list(pos_list: tf.Tensor) -> tf.Tensor:
    """ "Returns the indices that correspond to each node in the pos_list.
    Generates a range of values [0, n_nodes) = original node indices."""
    n_nodes = tf.shape(pos_list)[0]
    return tf.range(start=0, limit=n_nodes)


def gen_pos_indices_img(idx: np.ndarray, xy: np.ndarray, dim: int) -> np.ndarray:
    """Generates an image containing the integer indices of the node positions,
    at the corresponding (x,y) coordinates."""

    # placeholder data type must be bigger than uint8
    # -- if uint8, overflow can happen if there are more than 255 nodes
    img = np.zeros((dim, dim, 1)).astype(np.uint32)

    for i, (col, row) in enumerate(xy):
        img[row, col, :] = idx[i] + 1  # add one to avoid losing first index

    return img


def tf_pos_indices_image(i, xy, img_length):
    return tf.numpy_function(
        func=gen_pos_indices_img, inp=[i, xy, img_length], Tout=tf.uint32
    )


@tf.function
def unsorted_pos_list_from_image(node_pos_img: tf.Tensor) -> tf.Tensor:
    """Extracts the (unsorted) xy coordinates from the node_pos image."""
    node_pos_img = tf.cast(tf.squeeze(node_pos_img), tf.uint8)
    return tf.reverse(tf.where(node_pos_img), axis=[1])


def get_data_at_xy(matr: np.ndarray) -> np.ndarray:
    """Extracts data from a 2D matrix at the given (x,y) coordinate."""
    matr = matr.squeeze()
    rc = np.fliplr(pos_list_from_image(matr))
    return matr[rc[:, 0], rc[:, 1]] - 1


def rc_to_node_combo_img(
    rc1: np.ndarray, rc2: np.ndarray, dims: np.ndarray
) -> np.ndarray:
    """ "Converts a pair of (row, col) coordinates to a blank image
    with white dots corresponding to the given coordinates."""
    img = np.zeros(dims, dtype=np.int64)

    img[rc1[0], rc1[1]] = 1
    img[rc2[0], rc2[1]] = 1

    return img


@tf.function
def tf_rc_to_node_combo_img(rc1, rc2, img_dims):
    return tf.numpy_function(
        rc_to_node_combo_img, inp=(rc1, rc2, img_dims), Tout=tf.int64
    )


@tf.function
def tf_xy_to_node_combo_img(xy1, xy2, img_dims):
    rc1 = tf.reverse(xy1, axis=[-1])
    rc2 = tf.reverse(xy2, axis=[-1])
    return tf.numpy_function(
        rc_to_node_combo_img, inp=(rc1, rc2, img_dims), Tout=tf.int64
    )


def get_all_node_combinations(num_nodes: tf.Tensor) -> tf.Tensor:
    indices = tf.range(num_nodes)

    # meshgrid to create all possible pair combinations
    y, x = tf.meshgrid(indices, indices)
    x = tf.expand_dims(x, 2)
    y = tf.expand_dims(y, 2)
    z = tf.concat([x, y], axis=2)

    # extract the upper triangular part of the meshgrid
    all_combos = tf.constant([[0, 0]])  # placeholder
    for x in range(num_nodes - 1):
        # goes from 0 to n - 2
        aa = z[x + 1 :, x, :]
        all_combos = tf.concat([all_combos, aa], axis=0)
    all_combos = all_combos[1:, :]  # remove initial [0, 0] combination

    return all_combos


def get_reduced_node_combinations(
    all_combos: tf.Tensor, adj_matr: tf.Tensor, shuffle: bool = True
) -> tf.Tensor:
    """Returns a dataset of node combinations that have equal amounts of
    adjacent and non-adjacent node pairs. The dataset elements are
    in alternating order: adj > not adj > adj > not adj > ..."""
    adjacencies = [(pair, get_combo_adjacency(pair, adj_matr)) for pair in all_combos]

    # categorise according to adjacency
    adj_combos = [pair for (pair, adj) in adjacencies if adj == 1]
    not_adj_combos = [pair for (pair, adj) in adjacencies if adj == 0]

    if shuffle is True:
        np.random.shuffle(adj_combos)
        np.random.shuffle(not_adj_combos)

    # zip + nested list comprehension = interleaving while cutting off excess non_adj_nodes
    reduced_combos = [
        node
        for adj_and_non_adj in zip(adj_combos, not_adj_combos)
        for node in adj_and_non_adj
    ]

    return tf.stack(reduced_combos)


@tf.function
def node_pair_to_coords(
    pair: tf.Tensor, pos_list: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    xy1 = pos_list[pair[0], :]
    xy2 = pos_list[pair[1], :]

    rc1 = tf.reverse(xy1, axis=[0])
    rc2 = tf.reverse(xy2, axis=[0])

    return rc1, rc2


def get_combo_imgs(
    batch_combo: tf.Tensor,
    skel_img: tf.Tensor,
    pos_list: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    skel_img = tf.squeeze(skel_img)

    coords = [node_pair_to_coords(p, pos_list) for p in batch_combo]
    node_pair_imgs = [
        rc_to_node_combo_img(rc1, rc2, skel_img.shape) for (rc1, rc2) in coords
    ]

    return tf.stack(node_pair_imgs)


def get_combo_imgs_from_xy(
    combos_xy: tf.Tensor,
    img_dims: tf.Tensor,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(combos_xy).map(
        lambda x: (x[0], x[1], img_dims)
    )
    return ds.map(tf_xy_to_node_combo_img)


@tf.function
def get_combo_inputs(
    skel_img: tf.Tensor, node_pos_img: tf.Tensor, combos_xy: tf.Tensor
) -> tf.data.Dataset:
    img_dims = skel_img.shape
    combo_imgs = get_combo_imgs_from_xy(combos_xy, img_dims)
    num_neighbours = tf.cast(tf.shape(combos_xy)[0], tf.int64)

    return (
        combo_imgs.map(lambda x: (skel_img, node_pos_img, x))
        .batch(num_neighbours)
        .get_single_element()
    )


@tf.function
def get_combo_adjacency(pair: tf.Tensor, adj_matr: tf.Tensor) -> tf.Tensor:
    n1, n2 = pair[0], pair[1]
    return adj_matr[n1, n2]


@tf.function
def get_combo_path(
    pair: tf.Tensor, adjacency: tf.Tensor, pos_list: tf.Tensor, skel_img: tf.Tensor
) -> tf.Tensor:
    rc1, rc2 = node_pair_to_coords(pair, pos_list)
    row_indices = tf.sort([rc1[0], rc2[0]])
    col_indices = tf.sort([rc1[1], rc2[1]])

    img_section = skel_img[
        row_indices[0] : row_indices[1] + 1,
        col_indices[0] : col_indices[1] + 1,
    ]

    return tf.math.multiply(
        tf.cast(adjacency, tf.float32),
        tf.RaggedTensor.from_tensor(img_section),
    )


def repeat_to_match_dims(imgs: tf.Tensor, dims: List[tf.Tensor]) -> tf.Tensor:
    imgs_per_combo = [tf.stack([im for _ in range(n)]) for n, im in zip(dims, imgs)]
    imgs_in_batch = tf.concat(imgs_per_combo, axis=0)

    return imgs_in_batch


@tf.function
def rebatch(x: tf.data.Dataset, batch_size: int) -> tf.Tensor:
    """Converts tf.data.Dataset object to a tf.Tensor object"""
    return (
        x.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        .take(1)
        .get_single_element()
    )


@tf.function
def get_node_index(node_xy: tf.Tensor, pos_list: tf.Tensor) -> tf.Tensor:
    """Given xy coordinates of the node and a list of all node positions,
    returns the node id in the list."""
    node_id = tf.math.reduce_all(tf.equal(pos_list, node_xy), axis=1)
    return tf.squeeze(tf.where(node_id))


@tf.function
def get_node_neighbours(node_id: tf.Tensor, combos: tf.Tensor):
    """Returns the node combinations that contain node_id."""
    idx_contains_node = tf.math.reduce_any(tf.equal(combos, node_id), axis=1)
    idx_contains_node = tf.where(idx_contains_node)
    num_vals = tf.shape(idx_contains_node)[0]

    combo_ids = tf.reshape(
        tf.gather(combos, tf.squeeze(idx_contains_node)), (num_vals, 2)
    )

    return combo_ids


@tf.function
def distance(n1: tf.Tensor, n2: tf.Tensor, axis=None) -> tf.Tensor:
    """Calculates the distance between two nodes."""
    diff = tf.cast(n1 - n2, tf.float32)
    return tf.math.reduce_euclidean_norm(diff, axis=axis)


# @tf.function(experimental_relax_shapes=True)
def nearest_neighbours(node_id, num_neighbours, all_combos, pos_list):
    """Returns the nearest neighbours of the node at node_id."""
    # length might be < num_neighbours if all_combos has become sparse (due to nodes being discarded)
    combos = get_node_neighbours(node_id, all_combos)
    combos_xy = tf.gather(pos_list, combos)

    # if length is not ok, no need to calculate nearest neighbours
    if combos.shape[0] < num_neighbours:
        return combos

    # get the smallest distance
    combos_p1_xy, combos_p2_xy = combos_xy[:, 0, :], combos_xy[:, 1, :]
    distances = distance(combos_p1_xy, combos_p2_xy, axis=1)
    _, indices = tf.math.top_k(-1 * distances, k=num_neighbours)

    """
    neighbours_xy [k, 2, 2]
        axis 0: all the neighbours corresponding to the node
        axis 1: neighbour, node
        axis 2: x-values, y-values
    """
    # neighbours_xy = tf.gather(combos_xy, indices)

    return tf.gather(combos, indices)


def get_combo_id(combos: tf.Tensor, all_combos: tf.Tensor) -> tf.Tensor:
    """Returns index of combinations in the list of all possible combinations."""
    # [n_all_combos, n_combos, 2] -> [n_all_combos, n_combos]
    matches = tf.map_fn(
        lambda x: tf.math.equal(combos, x),
        elems=all_combos,
        fn_output_signature=tf.bool,
    )
    matches = tf.map_fn(lambda x: tf.math.reduce_all(x, axis=-1), elems=matches)
    matches = tf.map_fn(lambda x: tf.math.reduce_any(x, axis=-1), elems=matches)

    return tf.squeeze(tf.where(matches))


def discard_nodes(combos_id: tf.Tensor, all_combos: tf.Tensor) -> tf.Tensor:
    """Discard already searched node combinations.
    This function takes a long time to execute, possibly due to the del function (runs on CPU?).
    """
    # largest indices first
    idx_to_discard = get_combo_id(combos_id, all_combos)
    idx_to_discard = tf.reverse(idx_to_discard, axis=[0])

    new_combos = tf.unstack(all_combos)

    for i in idx_to_discard:
        del new_combos[i]

    # new dimensions: [n_all_combos - n_combos, 2]
    return tf.stack(new_combos)


def data_from_node_imgs(
    node_pos: tf.Tensor, degrees: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns lookup table consisting of node xy position and the corresponding degree."""
    pos_list_xy = sorted_pos_list_from_image(node_pos)
    pos_list_rc = tf.reverse(pos_list_xy, axis=[1])
    degrees_list = tf.gather_nd(indices=pos_list_rc, params=tf.squeeze(degrees))

    return pos_list_xy, degrees_list


@tf.function
def unique(tensor: tf.Tensor, axis: int, dtype=tf.int64) -> tf.Tensor:
    """Since tf.unique is currently for 1D tensors, use this for finding unique values across 2D or more."""
    return tf.numpy_function(
        lambda x: np.unique(x, axis=axis), inp=[tensor], Tout=dtype
    )


@tf.function
def remove_combo_subset(combos: tf.Tensor, subset: tf.Tensor) -> tf.Tensor:
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


@tf.function
def remove_combos_containing_nodes(combos: tf.Tensor, nodes: tf.Tensor) -> tf.Tensor:
    not_found = tf.map_fn(
        lambda node: tf.not_equal(combos, node),
        elems=nodes,
        fn_output_signature=tf.bool,
    )

    indices_not_found = tf.where(tf.reduce_all(not_found, axis=[0, 2]))[:, 0]

    return tf.gather(combos, indices_not_found)


@tf.function
def unique_nodes_from_combo(combos: tf.Tensor) -> Tuple[tf.Tensor, tf.RaggedTensor]:
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


@tf.function
def node_adjacencies(
    rows_in_combos: tf.RaggedTensor, adjacencies: tf.Tensor, adjacency_probs: tf.Tensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    # get the corresponding adjacencies for the unique nodes
    unique_adjacencies = tf.reduce_sum(tf.gather(adjacencies, rows_in_combos), axis=1)
    unique_adj_probs = tf.gather(adjacency_probs, rows_in_combos)

    return unique_adjacencies, unique_adj_probs


@tf.function
def adjacency_cases(
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


@tf.function
def new_adjacencies(args: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Returns new adjacency vector with <degree> of the highest probability entries."""
    probs, degree = args[0], args[1]
    max_prob_indices = tf.nn.top_k(probs, k=tf.cast(degree, tf.int32)).indices

    adj = tf.ones(tf.shape(max_prob_indices), dtype=tf.int32)
    shape = tf.shape(probs)

    new_adj = tf.scatter_nd(tf.expand_dims(max_prob_indices, axis=-1), adj, shape)
    return new_adj


@tf.function
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


@tf.function
def new_adjacency_probs(args: Tuple[tf.Tensor, tf.Tensor]):
    """Returns new adjacency probability vector with <degree> of the highest probability entries."""
    probs, degree = args[0], args[1]
    max_prob_indices = tf.nn.top_k(probs, k=tf.cast(degree, tf.int32)).indices

    adj_probs = tf.gather(probs, max_prob_indices)
    shape = tf.shape(adj_probs)

    new_adj_probs = tf.scatter_nd(
        tf.expand_dims(max_prob_indices, axis=-1), adj_probs, shape
    )
    return new_adj_probs


@tf.function
def get_new_adjacency_probs(node_adj_probs, node_degrees):
    """Returns new adjacency probability vector with <degree> of the highest probability entries."""
    return tf.map_fn(
        new_adjacency_probs,
        elems=(node_adj_probs, node_degrees),
        fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.float32),
    )


@tf.function
def update_bad_node_adjacencies(node_adj_probs, node_degrees):
    """Returns new adjacency vector and adjacency probability vector
    by taking only <degree> of the highest probability entries."""
    new_adj = get_new_adjacencies(node_adj_probs, node_degrees)
    new_adj_probs = get_new_adjacency_probs(node_adj_probs, node_degrees)

    return new_adj, new_adj_probs


@tf.function
def get_rc_ragged(ragged, flat_idx):
    """Converts the flat index to the [row, col] format."""
    row = tf.where(ragged.row_splits <= flat_idx)[-1, 0]

    row_split = tf.squeeze(tf.gather(ragged.row_splits, row))
    col = tf.squeeze(tf.math.floormod(flat_idx, row_split))

    return tf.stack((row, col))


@tf.function
def check_duplicate_combo_ids(node_rows: tf.RaggedTensor):
    unique_combo_ids, _, counts = tf.unique_with_counts(node_rows.flat_values)
    non_duplicates_ids = tf.gather(unique_combo_ids, tf.where(counts == 1))[:, 0]
    duplicates_ids = tf.gather(unique_combo_ids, tf.where(counts > 1))[:, 0]

    return non_duplicates_ids, duplicates_ids


@tf.function
def get_indices_of_combos_in_node_rows(
    node_rows: tf.RaggedTensor, combo_ids: tf.Tensor
) -> tf.Tensor:
    """Returns indices of flattened tensor"""
    empty_tensor = tf.equal(tf.size(combo_ids), 0)
    if empty_tensor:
        return combo_ids
    else:
        return tf.map_fn(
            lambda x: tf.where(node_rows.flat_values == x),
            elems=combo_ids,
        )


@tf.function
def empty_tensor(tensor: tf.Tensor) -> tf.bool:
    return tf.equal(tf.size(tensor), 0)


def get_combos_to_keep(
    combos: tf.Tensor, node_rows: tf.RaggedTensor, node_adjacencies: tf.RaggedTensor
):
    non_duplicate_combo_ids, duplicate_combo_ids = check_duplicate_combo_ids(node_rows)

    # for checking if empty
    exist_non_duplicates = tf.logical_not(empty_tensor(non_duplicate_combo_ids))
    exist_duplicates = tf.logical_not(empty_tensor(duplicate_combo_ids))
    if not (exist_duplicates and exist_non_duplicates):
        tf_empty = tf.cast(tf.reshape((), (0)), tf.int64)

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
        non_dup_adjs = tf_empty

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
        dup_adjs = tf_empty
        _dups_to_keep_idcs = tf_empty
        combo_ids = tf_empty
        adjacencies = tf_empty
        exist_nodes_to_discard = False

    # take non-discarded duplicate values
    combo_ids = tf.gather(duplicate_combo_ids, _dups_to_keep_idcs)
    adjacencies = (
        tf_empty
        if empty_tensor(dup_adjs)
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
