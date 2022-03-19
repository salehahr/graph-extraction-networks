from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import tensorflow as tf

import tools.combinations as combination_op
import tools.data as data_op
import tools.evaluate
import tools.neighbours
from tools.adj_matr import get_placeholders, get_update_function, preview
from tools.timer import timer

if TYPE_CHECKING:
    from model import EdgeNN


class AdjMatrPredictor:
    def __init__(self, model: EdgeNN, num_neighbours: int):
        self._model: EdgeNN = model
        self._init_num_neighbours = tf.constant(num_neighbours)
        self._num_neighbours = tf.Variable(
            initial_value=num_neighbours, trainable=False
        )

        # functions
        """Model prediction function, already traced."""
        self._predict_func: tf.types.experimental.ConcreteFunction = (
            tools.evaluate.get_edgenn_caller(model)
        )
        """Adjacency matrix update function, already traced."""
        self._update: tf.types.experimental.ConcreteFunction

        # placeholders
        self._A: tf.Variable

        """All initially available node pair combinations."""
        self._all_combos: tf.Tensor
        """Pool of combinations to choose from when generating neighbour combinations."""
        self._reduced_combos: tf.Tensor
        """Pair combinations based on number of neighbours set."""
        self._combos: tf.Tensor
        """Adjacency values corresponding to self._combos, as given by the model prediction."""
        self._adjacencies: tf.Tensor
        """Adjacency probability values corresponding to self._combos, as given by the model prediction."""
        self._adjacency_probs: tf.Tensor

        """Unique nodes fouund in self._combos."""
        self._nodes: tf.Tensor
        """Row indices where each node in self._nodes can be found in self._combos."""
        self._node_rows: tf.RaggedTensor
        """Summed adjacencies for each node in self._nodes, given the combinations self._combos."""
        self._node_adjacencies: tf.Tensor
        """Adjacency probabilities for each node in self._nodes, given the combinations self._combos."""
        self._node_adj_probs: tf.RaggedTensor
        """Degrees of each node in self._nodes."""
        self._node_degrees: tf.Tensor

        self._skel_img: tf.Tensor
        self._pos_list_xy: tf.Tensor

        # lookup tables
        self._adjacencies_lookup: tf.Variable
        self._degrees_lookup: tf.Variable

        # flags
        self._stop_iterate: tf.bool

    def _set_stop(self) -> None:
        """Sets flag to stop iterating."""
        self._stop_iterate = tf.constant(True)

    def _init_prediction(
        self, skel_img: tf.Tensor, node_pos: tf.Tensor, degrees: tf.Tensor
    ) -> None:
        """Initialises placeholders and flags before predicting."""

        # store skel img
        self._skel_img = skel_img
        self._num_neighbours.assign(self._init_num_neighbours)

        # derived data; constants/reference
        self._pos_list_xy, degrees_list, num_nodes = data_op.data_from_node_imgs(
            node_pos, degrees
        )
        degrees_list = tf.cast(degrees_list, tf.int64)

        self._all_combos = data_op.get_all_node_combinations(num_nodes)
        all_nodes = tf.expand_dims(tf.range(num_nodes, dtype=tf.int64), axis=-1)

        # iniitial/volatile; neighbours and prediction for neighbours
        self._reduced_combos = tf.identity(self._all_combos)
        self._combos = tools.neighbours.get_neighbours(
            self._num_neighbours.value(),
            self._reduced_combos,
            self._pos_list_xy,
            all_nodes,
        )
        self._adjacency_probs, self._adjacencies = self._get_predictions(node_pos)
        (
            self._nodes,
            self._node_rows,
            self._node_adjacencies,
            self._node_adj_probs,
            self._node_degrees,
        ) = combination_op.get_combo_nodes(
            self._combos,
            self._adjacencies,
            self._adjacency_probs,
            degrees_list,
        )

        # initialise lookup values
        self._adjacencies_lookup, self._degrees_lookup, self._A = get_placeholders(
            self._all_combos, degrees_list, num_nodes
        )
        self._update = get_update_function(self._A)

        # reset flags
        self._stop_iterate: tf.bool = tf.constant(False)

    @timer
    def predict(
        self,
        input_data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        do_preview: bool = False,
    ) -> Tuple[tf.Variable, tf.Tensor, tf.Tensor]:
        """Main prediction function."""

        self._init_prediction(*input_data)

        num_iters = tf.Variable(initial_value=0, trainable=False)
        while not self._stop_iterate:
            self._predict_ok_good()
            self._predict_bad()

            self._increase_neighbours()
            num_iters.assign_add(1)

        if do_preview:
            self._preview()

        return self._A, self._skel_img, self._pos_list_xy

    def _get_predictions(
        self,
        node_pos: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Obtains the model prediction on current neighbour node combinations.."""
        current_batch = data_op.get_combo_inputs(
            tf.squeeze(self._skel_img),
            tf.squeeze(node_pos),
            self._combos,
            self._pos_list_xy,
        )
        probabilities = self._predict_func(*current_batch)
        return tools.evaluate.classify(probabilities)

    def _increase_neighbours(self):
        """Doubles the number of neighbours to be searched and
        geerates new neighbour pair combinations.
        Sets stop flag if the newly generated combinations have
        the same length as the previous combinations.
        """
        # noinspection PyTypeChecker
        self._num_neighbours.assign(self._num_neighbours * 2)
        nodes_not_found = combination_op.nodes_not_found(self._degrees_lookup.value())

        len_old_combos = tf.shape(self._combos)[0]
        self._combos = tools.neighbours.get_neighbours(
            self._num_neighbours.value(),
            self._reduced_combos,
            self._pos_list_xy,
            nodes_not_found,
        )

        tf.cond(
            (len_old_combos == tf.shape(self._combos)[0]),
            true_fn=lambda: self._set_stop(),
            false_fn=lambda: None,
        )

    def _predict_ok_good(self) -> None:
        """Generates list of combos and nodes found that match the case
            NODE_ADJ <= NODE_DEGREES.
        Then updates the lookup tables and remove the found combinations and nodes
        from the placeholder lists of nodes and combinations.
        """
        case_combos, case_adj, nodes_found = combination_op.predict_ok_good(
            self._nodes,
            self._node_rows,
            self._node_adjacencies,
            self._node_adj_probs,
            self._node_degrees,
            self._combos,
            self._adjacencies,
            self._adjacencies_lookup,
            self._degrees_lookup,
        )
        self._update_after_prediction(case_combos, case_adj, nodes_found)

    def _predict_bad(self) -> None:
        """Generates list of combos and nodes found that match the case
            NODE_ADJ > NODE_DEGREES.
        Chooses the nodes and combinations such that the NEW_NODE_ADJ -  NODE_DEGREES >= 0.
        Then updates the lookup tables and remove the found combinations and nodes
        from the placeholder lists of nodes and combinations.
        """
        case_combos, case_adj, nodes_found = combination_op.predict_bad(
            self._nodes,
            self._node_rows,
            self._node_adjacencies,
            self._node_adj_probs,
            self._node_degrees,
            self._combos,
            self._adjacencies_lookup,
            self._degrees_lookup,
        )
        self._update_after_prediction(case_combos, case_adj, nodes_found)

    def _update_after_prediction(
        self, case_combos: tf.Tensor, case_adj: tf.Tensor, nodes_found: tf.Tensor
    ) -> None:
        """Updates adjacency matrix, then removes the combinations and nodes recently found
        from the lists of node pair combinations."""

        # update adjacency matrix
        tf.cond(
            tf.size(case_combos) > 0,
            true_fn=lambda: self._update(case_combos, case_adj, self._A),
            false_fn=lambda: None,
        )

        # update combos: remove found combinations
        self._reduced_combos = tf.cond(
            tf.size(case_combos) > 0,
            true_fn=lambda: combination_op.remove_combo_subset_from_all(
                self._reduced_combos, case_combos
            ),
            false_fn=lambda: self._reduced_combos,
        )
        self._combos, self._adjacencies, self._adjacency_probs = tf.cond(
            tf.size(case_combos) > 0,
            true_fn=lambda: combination_op.remove_combo_subset(
                self._combos, self._adjacencies, self._adjacency_probs, case_combos
            ),
            false_fn=lambda: (self._combos, self._adjacencies, self._adjacency_probs),
        )

        # update combos: remove found nodes
        (
            self._reduced_combos,
            self._combos,
            self._adjacencies,
            self._adjacency_probs,
        ) = tf.cond(
            tf.size(nodes_found) > 0,
            true_fn=lambda: combination_op.remove_nodes_found(
                self._reduced_combos,
                self._combos,
                self._adjacencies,
                self._adjacency_probs,
                nodes_found,
            ),
            false_fn=lambda: (
                self._reduced_combos,
                self._combos,
                self._adjacencies,
                self._adjacency_probs,
            ),
        )

        # set stop flag if pool of possible combos has reduced to zero
        tf.cond(
            tf.size(self._reduced_combos) == 0,
            true_fn=lambda: self._set_stop(),
            false_fn=lambda: None,
        )

        # update list of nodes from the remaining combos
        (
            self._nodes,
            self._node_rows,
            self._node_adjacencies,
            self._node_adj_probs,
            self._node_degrees,
        ) = combination_op.get_combo_nodes(
            self._combos,
            self._adjacencies,
            self._adjacency_probs,
            self._degrees_lookup.value(),
        )

    def _preview(self) -> None:
        """Plot the adjacency matrix."""
        preview(self._A, self._skel_img, self._pos_list_xy)
