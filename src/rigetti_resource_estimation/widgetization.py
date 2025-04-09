# Copyright 2022-2025 Rigetti & Co, LLC
#
# This Computer Software is developed under Agreement HR00112230006 between Rigetti & Co, LLC and
# the Defense Advanced Research Projects Agency (DARPA). Use, duplication, or disclosure is subject
# to the restrictions as stated in Agreement HR00112230006 between the Government and the Performer.
# This Computer Software is provided to the U.S. Government with Unlimited Rights; refer to LICENSE
# file for Data Rights Statements. Any opinions, findings, conclusions or recommendations expressed
# in this material are those of the author(s) and do not necessarily reflect the views of the DARPA.
#
# Use of this work other than as specifically authorized by the U.S. Government is licensed under
# the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""
**Module** ``rigetti_resource_estimation.widgetization``

Tools for dividing a circuit into smaller (possibly repeating) sub-circuits (called 'widgets').
"""
from __future__ import annotations
from typing import Dict, Any, Callable, Optional, Tuple
import cirq
import json
from functools import cached_property, lru_cache

from cirq import ops
from typing import List, Set

import networkx as nx
from pyLIQTR.utils.resource_analysis import estimate_resources
import rigetti_resource_estimation.gs_equivalence as gseq
import rigetti_resource_estimation.more_utils as utils
from dataclasses import dataclass, field

# This will likely get subsumed by 'decomposers' on another branch
#: This is the default alphabet that decompose_once or decompose_forever will stop decomposing at.
ALPHABET = ["I", "H", "S", "S**-1", "X", "Y", "Z", "T**-1", "T", "CZ", "CNOT", "ZPowGate"]
CircuitRepTuple = Tuple[cirq.Circuit, int]
PreFabDict = Dict[str, List[CircuitRepTuple]]


def small_enough_pyliqtr_re(max_gates: int = 10000, max_q: int = 100) -> Callable[[cirq.Circuit], bool]:
    """Create a function to determine if a circuit is small enough using pyLIQTR resource estimates."""

    def fn(circuit: cirq.Circuit) -> bool:
        """Decide if a circuit is 'small enough' according to pyLIQTR resource estimates."""
        estimate = estimate_resources(circuit)
        if estimate["T"] + estimate["Clifford"] <= max_gates and estimate["LogicalQubits"] <= max_q:
            return True
        return False

    return fn


class WidgetizationResult:
    """Class to store, process, and output results from widgetization.

    :param graph: a directed graph describing the nested structure of the operations in the circuit.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self._leaf_ids_to_widget_keys = {leaf.node_id: f"W-{i}" for i, leaf in enumerate(self.leaves)}
        self._leaf_ids_to_circuits = {leaf.node_id: leaf.circuit for i, leaf in enumerate(self.leaves)}

    @cached_property
    def leaves(self):
        """Return leaf nodes."""
        return [node for node, degree in self.graph.out_degree() if degree == 0]

    @cached_property
    def root_node(self):
        """Return the root node."""
        return [node for node, degree in self.graph.in_degree() if degree == 0][0]

    @lru_cache
    def _get_ordered_leaves_below_node(self, node):
        """Recursive function to return a list of (leaf_id, repetitions) below the passed node."""
        out_edges = [e for e in self.graph.out_edges(node, keys=True, data=True)]
        if len(out_edges) == 0:
            # leaf node
            return [(node.node_id, 1)]

        # order edges according to idx attribute
        # x[3] is the edge's data dictionary
        # x[:3] hold the edge's from node, to node, and key respectively
        ordered_edges = sorted(out_edges, key=lambda x: x[3]["idx"])
        nodes = []
        for edge in ordered_edges:
            weight = nx.get_edge_attributes(self.graph, name="weight")[edge[:3]]
            child_node = edge[1]  # edge's to node
            child_node_ids = self._get_ordered_leaves_below_node(child_node)
            if len(child_node_ids) == 1:
                nodes += [(node, node_reps * weight) for node, node_reps in child_node_ids]
            else:
                nodes += [(node, node_reps) for node, node_reps in child_node_ids] * weight
        return nodes

    @cached_property
    def circuit(self):
        """Reconstruct the full circuit from the graph."""
        circuit = cirq.Circuit()
        for leaf_node, reps in self.ordered_leaf_node_ids:
            for _ in range(reps):
                circuit.append(self._leaf_ids_to_circuits[leaf_node])
        return circuit

    @cached_property
    def ordered_leaf_node_ids(self):
        """Return a list of (leaf_id, repetitions) for the graph."""
        return self._get_ordered_leaves_below_node(self.root_node)

    @cached_property
    def widgets(self):
        """Return a map of leaf node ids to their respective circuits."""
        return {self._leaf_ids_to_widget_keys[leaf.node_id]: leaf.circuit for i, leaf in enumerate(self.leaves)}

    @lru_cache
    def _get_wands_below_node(self, node):
        """Recursive function to return widgets and stitches below the passed node."""
        out_edges = [e for e in self.graph.out_edges(node, keys=True, data=True)]
        if len(out_edges) == 0:
            # leaf node
            return utils.ItemAndInterfaceCounter.from_single_item(node.node_id)

        # order edges according to idx attribute
        # x[3] is the edge's data dictionary
        # x[:3] hold the edge's from node, to node, and key respectively
        ordered_edges = sorted(out_edges, key=lambda x: x[3]["idx"])
        result = None

        for edge in ordered_edges:
            weight = nx.get_edge_attributes(self.graph, name="weight")[edge[:3]]
            child_node = edge[1]  # edge's to node
            child_node_wands = self._get_wands_below_node(child_node)
            if result is None:
                result = weight * child_node_wands
            else:
                result += weight * child_node_wands
        return result

    @cached_property
    def wands(self):
        return self._get_wands_below_node(self.root_node)

    @cached_property
    def widgets_and_stitches(self):
        """Create a dictionary containing a count of widgets and a count of stitches between widgets."""
        wands = self.wands
        widget_counts = wands.items
        stitch_counts = wands.interfaces
        widgets = {self._leaf_ids_to_widget_keys[leaf.node_id]: widget_counts[leaf.node_id] for leaf in self.leaves}
        stitches = {
            (self._leaf_ids_to_widget_keys[leaf1_id], self._leaf_ids_to_widget_keys[leaf2_id]): stitch_counts[
                (leaf1_id, leaf2_id)
            ]
            for (leaf1_id, leaf2_id) in stitch_counts.keys()
        }
        return {"widgets": widgets, "stitches": stitches}


class QASMWidgetWriter:
    """A writer class to turn a WidgetizationResult and rewrite the widgets as QASM strings.

    This is useful for disk storage or backwards compatibility with Jabalizer and deprecated compilers.
    """

    def write(self, widget_results: WidgetizationResult) -> None:
        """Loop over leaves of circuit graph and return dictionary of widgets, qasms, etc.

        :param widget_results: the results from widgetization that will be rewritten as QASM strings.
        """
        qasm_outputs = []
        qasm_index_labels = []
        for widget_idx, widget in widget_results.widgets.items():
            qasm_outputs.append(widget._to_qasm_output())
            qasm_index_labels.append(widget_idx)

        all_qbs_appearing = widget_results.qbs_appearing
        qasm_strings = make_qasm_strings(qasm_outputs, all_qbs_appearing)
        qasms = {indx_label: qasm_str for indx_label, qasm_str in zip(qasm_index_labels, qasm_strings)}

        info = {"order": widget_results.order, "qasms": qasms, "qbs_appearing": all_qbs_appearing}
        return info


@dataclass
class CirqCircuitGraphNode:
    """A dataclass to act as nodes in a circuit decomposition graph.

    :param circuit: the circuit that the node represents.
    :param equiv_id_fn: function turning node's circuit into tuple description. Used for comparison of circuits.
    """

    circuit: cirq.Circuit
    equiv_id_fn: Optional[Callable[[cirq.Circuit], tuple[int]]] = field(  # type: ignore
        default_factory=lambda: CirqCircuitGraphNode._default_equiv_fn
    )

    @cached_property
    def node_id(self) -> int:
        return abs(hash(self.equiv_id_fn(self.circuit)))  # type: ignore

    @staticmethod
    def _default_equiv_fn(circuit):
        """This function would yield two circuits are the same if their gates and qubits are identical."""
        return tuple((op.gate, op.qubits) for op in circuit.all_operations())

    def __hash__(self):
        return self.node_id

    def __eq__(self, other):
        return self.node_id == other.node_id

    def __repr__(self):
        return "/".join([f"{op.gate.__class__.__name__},{op.qubits}" for op in self.circuit.all_operations()])


class CirqWidgetizer:
    """A class coordinating the widgetization of a circuit. Builds graph describing the nested structure of circuit."""

    def __init__(
        self,
        small_enough_fn: Callable[[cirq.Circuit], bool],
        gs_equiv_mapper: Optional[gseq.GraphStateEquivalenceMapper] = None,
        max_depth: int = 10,
        keep: Optional[Callable[[cirq.Circuit], bool]] = None,
        decomp_context: Optional[cirq.DecompositionContext] = None,
        pre_fab_dict: Optional[PreFabDict] = None,
    ):
        """
        :param small_enough_fn: a callable that determines if a circuit is below some size criteria.
        :param gs_equiv_mapper: A graph state equivalence class with a graph_state_equiv_tuple function that turns a
            circuit into a tuple. Two circuits with identical tuples imply the two circuits are graph state equivalent.
        :param max_depth: maximum gate decomposition level. Original passed subcircuit is a depth 0.
        :params keep: callable function to determine if a gate should NOT be decomposed.
        :param decomp_context: decomposition context to handle the qubit management during decomposition.
        :param pre_fab_dict: a dictionary describing a user defined decomposition for a gate.
        """
        self.small_enough_fn = small_enough_fn
        self.gs_equiv_mapper = gs_equiv_mapper
        self.max_depth = max_depth
        self.keep = keep
        self.decomp_context = decomp_context
        self.pre_fab_dict = pre_fab_dict or {}

    def widgetize(self, circuit: cirq.Circuit) -> WidgetizationResult:
        """Perform widgetization of the passed circuit.

        :param circuit: circuit to widgetize.
        """
        g = self.get_circuit_graph(circuit)
        return WidgetizationResult(g)

    def _op_to_circuit_rep_tuple(self, op):
        gate_key = op.gate.__class__.__name__
        prefab_fn = self.pre_fab_dict.get(gate_key, None)
        if prefab_fn is not None:
            subcircuits_and_reps = prefab_fn(op)
        else:
            # user predefined decomposition not present, use usual cirq based decomposition
            decomposed_ops = decompose_once(op, context=self.decomp_context)
            decomp = cirq.align_left(cirq.Circuit(decomposed_ops))

            # op does not decompose further
            if decomp == cirq.Circuit(op):
                return []

            # make a list of (subcircuit,reps) tuples.
            # Note: going operator-by-operator, each subcircuit will seem to appear once
            subcircuits_and_reps = [(cirq.Circuit(op), 1) for op in decomposed_ops]  # [(subcircuit, reps)]
        return subcircuits_and_reps

    def _build_subcircuit_graph(
        self,
        circuit: cirq.Circuit,
        g: nx.DiGraph,
        depth: int = 0,
    ) -> None:
        """Recursively make graph of circuit operation dependencies.

        :params circuit: circuit to make graph of subcircuit dependencies.
        :params g: current graph (updated recursively).
        :params depth: current level of decomposition.
        """
        if self.gs_equiv_mapper:
            equiv_id_fn = self.gs_equiv_mapper.graph_state_equiv_tuple
        else:
            equiv_id_fn = None
        new_node = CirqCircuitGraphNode(circuit=circuit, equiv_id_fn=equiv_id_fn)  # type: ignore
        if new_node in g:
            # We already visited this node.
            return

        # Add new node to the graph.
        g.add_node(new_node)

        # # Base case 1: This node is requested by the user to be a leaf node via the `keep` parameter.
        # # Not yet implemented. Likely will use decomposers when merged
        # if self.keep(subcircuit):
        #     return

        # Base case 2: Max depth exceeded
        if self.max_depth is not None and depth >= self.max_depth:
            return

        # Base case 3: Subcircuit is small enough to compile
        if self.small_enough_fn(circuit):
            return

        # Decompose
        ops = list(circuit.all_operations())
        if len(ops) > 1:
            # circuit is made up of multiple moments, so this 'decomposition' just splits them apart
            subcircuits_and_reps = [(cirq.Circuit(op), 1) for op in ops]
        elif len(ops) == 1:
            subcircuits_and_reps = self._op_to_circuit_rep_tuple(ops[0])
        else:
            return

        for idx, (subcircuit, reps) in enumerate(subcircuits_and_reps):
            # Quite important: we do the recursive call first before adding in the edges.
            # Otherwise, adding the edge would mark the callee node as already-visited by
            # virtue of it being added to the graph with the `g.add_edge` call.

            # Do the recursive step, which will continue to mutate `g`
            self._build_subcircuit_graph(subcircuit, g, depth + 1)

            # Update edge in `g`
            subcircuit_node = CirqCircuitGraphNode(circuit=subcircuit, equiv_id_fn=equiv_id_fn)  # type: ignore
            g.add_edge(new_node, subcircuit_node, weight=reps, idx=idx)  # idx indexes order the subcircuits appear in
        return

    def get_circuit_graph(self, circuit: cirq.Circuit) -> nx.DiGraph:
        """Create DAG of nested gate definitions from RRECircuit."""
        g = nx.MultiDiGraph()
        if len(circuit) > 1:
            circuit = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(circuit)))
        self._build_subcircuit_graph(circuit, g=g, depth=0)
        return g


def decompose_once(
    op: cirq.ops.Operation, alphabet: List[str] = ALPHABET, context=Optional[cirq.DecompositionContext]
) -> cirq.ops.Operation:
    """Decompose an operation one time, provided it is not in the alphabet.

    :param op: operation to decompose.
    :param alphabet: list of strings to compare gate names against. Ops matching these strings will NOT be decomposed.
    :param context: decomposition context to handle the qubit management during decomposition.
    """
    if gseq.gate_class(op) in alphabet and not isinstance(op, cirq.CircuitOperation):
        return op
    return cirq.decompose_once(op, default=op, context=context)  # type: ignore


def make_qasm_strings(qasm_outputs: List[cirq.QasmOutput], all_qbs_appearing: Set[cirq.Qid]) -> List[str]:
    """Creates a list of N QASM strings from an input of N QasmOutput objects.

    :param qasm_outputs: a list of QasmOutput objects to generate strings from. This includes relabelling qubits.
    :param all_qbs_appearing: a set of all qubits that will be appearing in the QASM strings.
    """
    qasm_strings = []

    # Determine ordering of qubits
    order = cirq.QubitOrder.DEFAULT
    ordered_list = ops.QubitOrder.as_qubit_order(order).order_for(all_qbs_appearing)

    # Generate qubit map and list of LineQubits to replace all qubits by
    qb_qasm_map = {qubit: f"q[{i}]" for i, qubit in enumerate(ordered_list)}
    lineqbs = cirq.LineQubit.range(len(ordered_list))

    # Generate strings
    for qasm_output in qasm_outputs:
        qasm_output.qubits = lineqbs  # type: ignore
        qasm_output.args.qubit_id_map = qb_qasm_map
        qasm_strings.append(str(qasm_output))
    return qasm_strings


def to_json_file(info: Dict[str, Any], fname: str) -> None:
    """Write info dictionary to json file."""
    if "widgets" in info:
        info.pop("widgets")
    with open(fname, "w", encoding="utf8") as f:
        json.dump(info, f)
    print(f"Widgets and info written to {fname}")
