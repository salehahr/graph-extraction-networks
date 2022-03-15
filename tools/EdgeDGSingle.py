from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple

import tensorflow as tf

from tools import data as data_op
from tools import plots, tables
from tools.adj_matr import update_adj_matr
from tools.postprocessing import tf_classify

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from tensorflow.lookup.experimental import DenseHashTable

    from model import EdgeNN


class AdjClassification(Enum):
    GOOD = 0
    OK = 1
    BAD = 2


# noinspection PyUnreachableCode,PyUnusedLocal
def check_degree_differences(
    case: AdjClassification, degrees_difference: tf.Tensor
) -> None:
    """Checks the the difference = node_degrees - node_adjacencies corresponds to the case
    currently being handled."""
    if not __debug__:
        return

    if case == AdjClassification.GOOD:
        assert tf.reduce_all(tf.equal(degrees_difference, 0))
    elif case == AdjClassification.OK:
        assert tf.reduce_all(tf.greater(degrees_difference, 0))
    elif case == AdjClassification.BAD:
        assert tf.reduce_all(tf.greater_equal(degrees_difference, 0))


class EdgeDGSingle(tf.keras.utils.Sequence):
    """
    Generates node combinations for a single skeletonised image.
    """

    def __init__(
        self,
        num_neighbours: int,
        skel_img: tf.Tensor,
        node_pos: tf.Tensor,
        degrees: tf.Tensor,
    ):
        self.skel_img = tf.squeeze(skel_img)
        self.node_pos = tf.squeeze(node_pos)
        self.degrees = tf.squeeze(degrees)

        # derived data
        derived_data = data_op.data_from_node_imgs(node_pos, degrees)
        self.pos_list_xy = derived_data[0]
        self.degrees_list = derived_data[1]

        self.num_nodes = tf.shape(self.pos_list_xy)[0]
        self.all_nodes = tf.range(0, self.num_nodes, dtype=tf.int64)

        # placeholder
        self.adj_matr = tf.zeros((self.num_nodes, self.num_nodes), dtype=tf.uint8)

        # all combinations (non-volatile)
        all_combos = data_op.get_all_node_combinations(self.num_nodes)
        self.all_combos = tf.cast(all_combos, tf.int64)

        # lookup tables -- N: all nodes, NN: all combinations
        self.tab_NN_ids = None
        self.tab_NN_adjacencies = None
        self.tab_N_degrees = None

        self._get_lookup_tables()

        # volatile
        """Only node combinations where no adjacencies have been set yet."""
        self._reduced_combos = tf.cast(all_combos, tf.int64)
        """Reduced node combinations limited to the nearest neighbours."""
        self._combos = None

        self.num_neighbours = num_neighbours
        self.update_neighbours(num_neighbours)

        # batching
        self.batch_size = self.num_combos  # initial batch size

    def _get_lookup_tables(self) -> None:
        """Initialise lookup tables."""
        tables_ = tables.init_lookup_tables()

        # placeholders
        self.tab_NN_ids = tables_["NN"]["ids"]
        self.tab_NN_adjacencies = tables_["NN"]["adjacency"]
        self.tab_N_degrees = tables_["N"]["degrees"]

        # initialise values
        num_combos = tf.shape(self.all_combos)[0]
        self.tab_N_degrees.insert(self.all_nodes, tf.cast(self.degrees_list, tf.int64))
        self.tab_NN_ids.insert(self.all_combos, tf.range(num_combos, dtype=tf.int64))

    def update_neighbours(self, num_neighbours: int) -> None:
        """Increases number of radius to search for and finds the corresponding
        node combinations."""
        self.num_neighbours = num_neighbours
        self._combos = self._get_neighbours()

    def _get_neighbours(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Obtains k nearest neighbours from the available node combinations."""
        # make RaggedTensors to deal with cases when there are not enough neighbours
        combos = tf.map_fn(
            lambda x: data_op.nearest_neighbours(
                x, self.num_neighbours, self._reduced_combos, self.pos_list_xy
            ),
            elems=self.nodes_not_found_yet,
            fn_output_signature=tf.RaggedTensorSpec(
                shape=[None, 2], ragged_rank=0, dtype=tf.int64
            ),
        )
        # combos = tf.concat(tf.unstack(combos, axis=1), axis=0)
        combos = combos.flat_values

        # flatten [n_nodes], [n_neighbours] -> [n_nodes x n_neighbours]
        # num_elems = tf.reduce_prod(tf.shape(combos)[0:2])
        # combos = tf.stack(tf.reshape(combos, (num_elems, 2)), axis=0)
        # combos_xy = tf.stack(tf.reshape(combos_xy, (num_elems, 2, 2)), axis=0)

        """
        combos [n_nodes, n_neighbours, 2(p1, p2)]
            combos[0] = ids of n_neighbours pairs
        combos_xy [n_nodes, n_neighbours, 2(p1, p2), 2(x, y)]
        """

        return combos

    def _update_combos(
        self, combos: tf.Tensor, combos_to_remove: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        # combos: combinations where adjacencies have just been set
        if combos_to_remove is not None:
            combos = data_op.remove_combo_subset(combos, combos_to_remove)

        # exclude found nodes to give combos containing nodes that have not been found yet
        combos = data_op.remove_combos_containing_nodes(combos, self.nodes_found)

        return combos

    def _update_neighbour_combos(self, remove: Optional[tf.Tensor] = None) -> None:
        self._combos = self._update_combos(self._combos, combos_to_remove=remove)

    def _update_reduced_combos(self, remove: Optional[tf.Tensor] = None) -> None:
        self._reduced_combos = self._update_combos(
            self._reduced_combos, combos_to_remove=remove
        )

    @property
    def num_combos(self):
        """Number of nearest neighbour combinations."""
        return len(self._combos)

    @property
    def nodes_found(self) -> tf.Tensor:
        """List of nodes for which exactly <degree> edges have been found."""
        return tf.where(self.all_degrees_n == 0)[:, 0]

    @property
    def nodes_not_found_yet(self) -> tf.Tensor:
        """List of nodes for which not enough edges have been found yet."""
        return tf.where(self.all_degrees_n != 0)[:, 0]

    @property
    def checked_all_nodes(self) -> tf.bool:
        """Check whether node degrees has been satisfied, i.e. if degrees have been set to zero."""
        return tf.reduce_all(tf.equal(self.all_degrees_n, 0))

    @property
    def checked_all_combinations(self) -> tf.bool:
        """Check whether all node combinations have been iterated through (-1 if not)."""
        return tf.reduce_all(tf.not_equal(self.all_adjacencies_nn, -1))

    @property
    def all_degrees_n(self):
        """List of all degrees corresponding to each node."""
        return self.tab_N_degrees[self.all_nodes]

    @property
    def all_adjacencies_nn(self):
        """List of all the adjancencies corresponding to each node combination."""
        return self.tab_NN_adjacencies[self.all_combos]

    def __len__(self) -> int:
        """How many iteration steps are needed to traverse all the neighbour node combimations?"""
        # ceil instead of floor division
        if self.batch_size == 0:
            return 0
        else:
            return -(self.num_combos // -self.batch_size)

    def __getitem__(self, i: int) -> tf.data.Dataset:
        """Gets batch of nearest neighbour combinations in [x, y] coordinates,
        converts it to the necessary format that EdgeNN expects."""
        combos = self._get_unique_combos_in_batch(i)
        return self._get_combo_inputs(combos)

    def _get_unique_combos_in_batch(self, i: int) -> tf.Tensor:
        """Gets nearest neighbour combinations, without duplicates"""
        combos = self._combos[
            i * self.batch_size : i * self.batch_size + self.batch_size
        ]
        return data_op.unique(combos, axis=0, dtype=tf.int64)

    def _get_combo_inputs(self, combos: tf.Tensor) -> tf.data.Dataset:
        """Returns the current node combinations the format required by the model for prediction."""
        combos_xy = tf.gather(self.pos_list_xy, combos)
        return data_op.get_combo_inputs(self.skel_img, self.node_pos, combos_xy)

    def update_adjacencies(self, i: int, model: EdgeNN) -> None:
        # current batch of nearest neighbour node combinations.
        # volatile: this gets overwritten in the for loop below
        self._combos = self._get_unique_combos_in_batch(i)

        # these lookup tables are for easy getting of values later on
        tab_adjacency_probs, tab_adjacencies = self._get_batch_lookup_tables(model)

        # Fallunterscheidung -- need to discard combinations where adjacencies found > degrees
        # deal with GOOD nodes first -> OK nodes -> BAD nodes/combos
        for case in AdjClassification:
            # get adjacency lists corresponding to the node combinations
            adjacency_probs = tab_adjacency_probs[self._combos]
            adjacencies = tab_adjacencies[self._combos]

            # get list of nodes, without duplicates, which are contained in the neighbours node combinataions.
            nodes, node_rows = data_op.unique_nodes_from_combo(self._combos)
            node_adjacencies, node_adj_probs = data_op.node_adjacencies(
                node_rows, adjacencies, adjacency_probs
            )
            node_degrees = self.tab_N_degrees[nodes]

            # set comparison type
            if case == AdjClassification.GOOD:
                comparison = node_adjacencies == node_degrees
            elif case == AdjClassification.OK:
                comparison = node_adjacencies < node_degrees
            elif case == AdjClassification.BAD:
                comparison = node_adjacencies > node_degrees

            # only get nodes which match current comparison type
            (
                nodes,
                node_rows,
                node_adjacencies,
                node_adj_probs,
            ) = data_op.adjacency_cases(
                comparison, nodes, node_rows, node_adjacencies, node_adj_probs
            )
            node_degrees = self.tab_N_degrees[nodes]

            # skip this case if no nodes found
            if data_op.is_empty_tensor(nodes):
                continue

            # first obtain case_combos (neighbour node combinations which match the case)
            # and their corresponding adjacencies
            if case is not AdjClassification.BAD:
                case_combos, adjacencies = self._update_adjacencies_good_ok_nodes(
                    node_rows, tab_adjacencies
                )
            # this step additionally updates node_adjacencies
            # (values get capped so that num of node_adjacencies <= degrees per node)
            else:
                (
                    case_combos,
                    adjacencies,
                    node_adjacencies,
                ) = self._update_adjacencies_bad_nodes(
                    node_rows, node_adj_probs, node_degrees
                )

            # skip this case if no case_combos found
            if data_op.is_empty_tensor(case_combos):
                continue

            # update adjacency matrix
            self._update_adj_matr(case_combos, adjacencies)
            # noinspection PyUnreachableCode
            if __debug__:
                self.preview(
                    title=f"New edges in {case}",
                    blank=True,
                    combos=case_combos,
                    adjacencies=adjacencies,
                )

            # decrease degrees of nodes in the combos -- once degrees reaches 0, the node is 'found'
            degrees_difference = node_degrees - node_adjacencies
            self._update_found_nodes(case, nodes, degrees_difference)

            # remove case_combos (combos where adj has just been set),
            # as well as combos containing found nodes,
            # from self._combos (only neighbours) and
            # from self._reduced_combos (supposed to be only combos with unset adjacencies)
            self._update_neighbour_combos(remove=case_combos)
            self._update_reduced_combos(remove=case_combos)

            # won't work for the next case if all the neighbours have been examined/removed
            if self.combos_is_empty:
                break

    @property
    def combos_is_empty(self) -> tf.bool:
        """Are there no neighbours left?"""
        return data_op.is_empty_tensor(self._combos)

    def _update_adjacencies_good_ok_nodes(
        self, node_rows: tf.RaggedTensor, tab_adjacencies: DenseHashTable
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Used for nodes for which exactly/less than <degree> number of node combinations are detected."""
        unique_combo_ids = tf.unique(node_rows.flat_values).y
        case_combos = tf.gather(self._combos, unique_combo_ids)
        adjacencies = tab_adjacencies[case_combos]

        # update lookup table
        self.tab_NN_adjacencies.insert(case_combos, adjacencies)

        return case_combos, adjacencies

    def _update_adjacencies_bad_nodes(
        self,
        node_rows: tf.RaggedTensor,
        node_adj_probs: tf.RaggedTensor,
        node_degrees: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Bad nodes are nodes for which more adjacent neighbours are detected than there are degrees.
        This function deals with this by capping the number of adjacencies found according to the
        degrees and by taking the max. adjacency probabilities.

        This set of capped adjacencies is then checked to make sure that the capping did not cause
        conflicts between two nodes in a particular combination.

        Only after this does the adjacencies table get updated.
        """
        # cap adjacencies based on degree
        node_adjacencies_capped = data_op.get_new_adjacencies(
            node_adj_probs, node_degrees
        )

        # select which combos to proceed with (discard if duplicate values are problematic)
        case_combos, adjacencies, node_adjacencies = data_op.get_combos_to_keep(
            self._combos, node_rows, node_adjacencies_capped
        )

        # update lookup table
        self.tab_NN_adjacencies.insert(case_combos, adjacencies)

        return case_combos, adjacencies, node_adjacencies

    def _update_found_nodes(
        self, case: AdjClassification, nodes: tf.Tensor, degrees: tf.Tensor
    ) -> None:
        """Updates degrees table with the new degrees. Found nodes are updated with an 0."""
        check_degree_differences(case, degrees)
        self.tab_N_degrees.insert(nodes, degrees)

    def _get_batch_lookup_tables(
        self, model: EdgeNN
    ) -> Tuple[DenseHashTable, DenseHashTable]:
        """Performs prediction on batch and generate adjacency lookups."""
        adjacency_probs, adjacencies = self._get_predictions(model, self._combos)
        tab_adjacency_probs, tab_adjacencies = tables.combo_lookup_tables(
            self._combos, adjacency_probs, adjacencies
        )
        return tab_adjacency_probs, tab_adjacencies

    def _get_predictions(
        self, model: EdgeNN, combos: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Performs model prediction on current batch of node combinations."""
        current_batch = self._get_combo_inputs(combos)

        adjacency_probs = tf.squeeze(tf.constant(model.predict(current_batch)))
        adjacencies = tf_classify(adjacency_probs)

        return adjacency_probs, adjacencies

    def _update_adj_matr(self, combos: tf.Tensor, adjacencies: tf.Tensor) -> None:
        """Updates the stored adjacency matrix with the given node pair combinations and their
        corresponding adjacencies."""
        self.adj_matr = update_adj_matr(self.adj_matr, adjacencies, combos)

    def preview(
        self,
        combos: Optional[tf.Tensor] = None,
        adjacencies: Optional[tf.Tensor] = None,
        title: Optional[str] = None,
        blank: bool = False,
    ) -> None:
        """Previews the predicted adjacency matrix as a grpah overlaid over the skeletonised image."""

        # pnly preview the most recent change to the adjacency matrix
        if blank:
            adj_matr = tf.zeros((self.num_nodes, self.num_nodes), dtype=tf.uint8)
            adj_matr = update_adj_matr(adj_matr, adjacencies, combos).numpy()
        # preview the complete adjacency matrix
        else:
            if adjacencies is not None and combos is not None:
                self._update_adj_matr(combos, adjacencies)
            adj_matr = self.adj_matr.numpy()

        plots.plot_adj_matr(
            self.skel_img.numpy(), self.pos_list_xy.numpy(), adj_matr, title=title
        )
