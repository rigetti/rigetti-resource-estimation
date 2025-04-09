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

"""Test for `rigetti_resource_estimation` widgetization module."""
import cirq
import cirq.testing as ct
import rigetti_resource_estimation.gs_equivalence as gseq
from rigetti_resource_estimation import widgetization
from rigetti_resource_estimation.widgetization import CirqCircuitGraphNode, CirqWidgetizer
from unittest.mock import patch
import pytest
import networkx as nx
from typing import Callable


def small_enough(max_q) -> Callable[[cirq.Circuit], bool]:
    """Create an example function to determine if a circuit is small enough based on number of qubits."""

    def fn(circuit: cirq.Circuit) -> bool:
        """Decide if a circuit is 'small enough'."""
        num_qubits = len(circuit.all_qubits())
        if num_qubits < max_q:
            return True
        return False

    return fn


class MySmallerGate(cirq.Gate):
    """A gate using 5 qubits."""

    def __init__(self):
        super(MySmallerGate, self)

    def _num_qubits_(self):
        return 5

    def _decompose_(self, qubits):
        """Decomposition is arbitrary."""
        for i in range(len(qubits) - 1):
            a, b = qubits[i : i + 2]
            yield cirq.CNOT(a, b)
            yield cirq.CNOT(b, a)
            yield cirq.CNOT(a, b)


class MyGate(cirq.Gate):
    """A gate using 10 qubits."""

    def __init__(self):
        super(MyGate, self)
        self.example_params = [5, 10, 7, 3]

    def _num_qubits_(self):
        return 10

    def _decompose_(self, qubits):
        """Decomposition is arbitrary."""
        for i in range(len(qubits) - 1):
            a, b = qubits[i : i + 2]
            yield cirq.CNOT(a, b)
            yield cirq.CNOT(b, a)
            yield cirq.CNOT(a, b)


@pytest.fixture
def qbs():
    """Returns 10 line qubits for use in tests."""
    return cirq.LineQubit.range(10)


@pytest.fixture
def prefab_dict1(qbs):
    """Return a customized breakdown of the MyGate gate."""

    def prefab1_fn(op):
        """Reads off the example params from the operation's gate, and uses it to create a prefab list."""
        p1, p2, p3, p4 = op.gate.example_params
        return [
            (cirq.Circuit([cirq.X(qbs[0])]), p1),
            (cirq.Circuit(cirq.CNOT(qbs[0], qbs[1])), p2),
            (cirq.Circuit(MySmallerGate().on(*qbs[:5])), p3),
            (cirq.Circuit(cirq.Y(qbs[1])), p4),
        ]

    return {"MyGate": prefab1_fn}


@pytest.fixture
def prefab_dict2(prefab_dict1, qbs):
    """Return a customized breakdown of the MyGate and MySmallerGate gates."""

    def prefab2_fn(op):
        return [(cirq.Circuit(cirq.Z(qbs[1])), 2), (cirq.Circuit(cirq.H(qbs[3])), 11)]

    prefab_dict2 = prefab_dict1.copy()
    prefab_dict2["MySmallerGate"] = prefab2_fn
    return prefab_dict2


@pytest.fixture
def prefab_dict3(qbs):
    """Return a customized breakdown of the MyGate gate."""

    def prefab3a_fn(op):
        multi_moment_circuit = cirq.Circuit(
            [cirq.H(qbs[0]).with_tags(4), MySmallerGate().on(*qbs[:5]).with_tags(10), cirq.X(qbs[0]).with_tags(11)]
        )
        return [(multi_moment_circuit, 7)]

    def prefab3b_fn(op):
        return [(cirq.Circuit(cirq.Z(qbs[1])), 2), (cirq.Circuit(cirq.H(qbs[3])), 11)]

    return {"MyGate": prefab3a_fn, "MySmallerGate": prefab3b_fn}


@pytest.fixture
def circuit(qbs):
    """Return a simple, single, custom gate circuit using 10 qubits."""
    return cirq.Circuit([MyGate().on(*qbs)])


def get_widgetized_graph(circuit, maxq, pre_fab_dict=None, gs_equiv_mapper=gseq.GraphStateEquivalenceMapper()):
    """Return a widgetized graph for a given circuit, pre_fab_dict, and gs equiv mapper."""
    small_enough_fn = small_enough(max_q=maxq)
    widgetizer = CirqWidgetizer(
        small_enough_fn=small_enough_fn, gs_equiv_mapper=gs_equiv_mapper, pre_fab_dict=pre_fab_dict
    )
    widgetizer_result = widgetizer.widgetize(circuit)
    return widgetizer_result.graph


@pytest.fixture
def get_circuit_equiv_map_small_fn():
    circuit = ct.random_circuit(qubits=10, n_moments=100, op_density=0.7)
    gs_equiv_mapper = gseq.GraphStateEquivalenceMapper()
    small_enough_fn = widgetization.small_enough_pyliqtr_re(max_gates=2, max_q=5)  # arbitrary
    return circuit, gs_equiv_mapper, small_enough_fn


@pytest.fixture
def example_circuit():
    """Creates a circuit that produces the example_graph below.

    Graph should look like (numbers are circuit/op tags, NOT node_ids):
                 9999
       |----------|------------|
      21          15           21
    |-|-|      | ---- |      *-*-*
    1 2 3      21     1      1 2 3
             *-*-*
             1 2 3
    Recorded edges are indicated by |, skipped edges (already seen nodes) indicated by *.
    Any circuit/op with a tag larger than 3 will be decomposed.
    Final list of leaves at root node (9999) should be (left to right):
    [1,2,3,1,2,3,1,1,2,3]. There should be 6 unique nodes: 9999, 21, 15, 1, 2, 3.
    There should be 7 unique edges: (9999,21), (9999,15), (21,1), (21,2), (21,3), (15,21), (15,1).
    """
    # create random circuits
    circuits = [ct.random_circuit(qubits=1, n_moments=3, op_density=0.7, random_state=42) for _ in range(3)]

    # smallest circuits
    op1 = cirq.CircuitOperation(cirq.FrozenCircuit(circuits[0])).with_tags(1)
    op2 = cirq.CircuitOperation(cirq.FrozenCircuit(circuits[1])).with_tags(2)
    op3 = cirq.CircuitOperation(cirq.FrozenCircuit(circuits[2])).with_tags(3)

    # create circuit with tag 21
    c = cirq.Circuit()
    c.append([op1, op2, op3])
    op3 = cirq.CircuitOperation(cirq.FrozenCircuit(c)).with_tags(21)

    # create circuit with tag 15
    c2 = cirq.Circuit()
    c2.append([op3, op1])
    op4 = cirq.CircuitOperation(cirq.FrozenCircuit(c2)).with_tags(15)

    # create circuit with tag 9999
    circuit = cirq.Circuit()
    circuit.append([op3, op4, op3])
    circuit = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(circuit)).with_tags(9999))
    return circuit


@pytest.fixture
def example_graph():
    """Creates the following graph for testing purposes.

    node_id shown with variable name in parenthesis. node_3, node_4, node_5 ids are 111, 222, 333 respectively.
    Recorded edges and weights are indicated by |int, skipped edges (already seen nodes) indicated by *.


                             123(node_0)
               |1-----------------|2------------------------|3
         456(node_1)           789(node_2)              456(node_1)
      |5---|7---|11         |13-----------|17            *---*---*
       111 222 333        456(node_1)    111            111 222 333
                         *---*---*
                        111 222 333

    leaf ids: [111, 222, 333]
    root id: 123
    """
    # Fake circuit ids
    circuit_node_ids = [123, 456, 789, 111, 222, 333]

    # example subcircuits (from a decomposition process, for example)
    circuits = [ct.random_circuit(qubits=10, n_moments=100, op_density=0.7) for _ in range(len(circuit_node_ids))]

    # Create graph nodes, providing id function that just returns a their hard coded id rather than a has of a circuit
    # tuple
    node_0 = CirqCircuitGraphNode(circuits[0], equiv_id_fn=lambda x: 123)
    node_1 = CirqCircuitGraphNode(circuits[1], equiv_id_fn=lambda x: 456)
    node_2 = CirqCircuitGraphNode(circuits[2], equiv_id_fn=lambda x: 789)
    node_3 = CirqCircuitGraphNode(circuits[3], equiv_id_fn=lambda x: 111)
    node_4 = CirqCircuitGraphNode(circuits[4], equiv_id_fn=lambda x: 222)
    node_5 = CirqCircuitGraphNode(circuits[5], equiv_id_fn=lambda x: 333)

    # Create edges
    edges = [
        (node_0, node_1, {"idx": 0, "weight": 1}),
        (node_0, node_2, {"idx": 1, "weight": 2}),
        (node_0, node_1, {"idx": 2, "weight": 3}),
    ]  # edges from root node
    edges += [
        (node_1, node_3, {"idx": 0, "weight": 5}),
        (node_1, node_4, {"idx": 1, "weight": 7}),
        (node_1, node_5, {"idx": 2, "weight": 11}),
        (node_2, node_1, {"idx": 0, "weight": 13}),
        (node_2, node_3, {"idx": 1, "weight": 17}),
    ]  # edges from next layer

    # Create graph
    graph = nx.MultiDiGraph()
    graph.add_edges_from(edges)
    leaf_nodes = [node_3, node_4, node_5]
    root_node = node_0
    return graph, leaf_nodes, root_node


def compare_widgetized_circuit_to_original(
    test_circuit: cirq.Circuit, gs_equiv_mapper: gseq.GraphStateEquivalenceMapper
) -> None:
    """This is a helper method to compare if widgetization/reconstruction of a circuit, is gs equiv to the original.

    :param test_circuit: circuit to widgetize and compare against.
    :param gs_equiv_mapper: mapper to map the circuits into graph state equivalence tuples.
    """

    # Widgetize
    small_enough_fn = small_enough(max_q=2)
    widgetizer = widgetization.CirqWidgetizer(small_enough_fn=small_enough_fn, gs_equiv_mapper=gs_equiv_mapper)
    result = widgetizer.widgetize(test_circuit)

    # Reconstruct from widgets
    reconstructed_circuit = cirq.align_left(cirq.Circuit(cirq.decompose(result.circuit)))

    decomposed_test_circuit = cirq.align_left(cirq.Circuit(cirq.decompose(test_circuit)))

    should_be_gs_labels = gs_equiv_mapper.graph_state_equiv_tuple(decomposed_test_circuit)
    actual_gs_labels = gs_equiv_mapper.graph_state_equiv_tuple(reconstructed_circuit)

    assert should_be_gs_labels == actual_gs_labels


def test_widgetization():
    """Should reconstructed circuit should have the same graph state equivalence tuple as original circuit."""
    # Graph state equivalence definition
    gs_equiv_mapper = gseq.GraphStateEquivalenceMapper()

    rc = ct.random_circuit(qubits=3, n_moments=100, op_density=0.7)

    compare_widgetized_circuit_to_original(rc, gs_equiv_mapper)


def test_widgetization_uses_default_qubit_manager(get_circuit_equiv_map_small_fn):
    """Context used by decompose_once should employ a different cirq.SimpleQubitManager for each decomposition."""
    circuit, gs_equiv_mapper, small_enough_fn = get_circuit_equiv_map_small_fn
    context = None
    with patch("cirq.protocols.decompose_protocol.decompose_once") as mock_decomp:
        # use small enough function forcing at least one decomposition to test which qubit manager is being used
        widgetizer = widgetization.CirqWidgetizer(
            small_enough_fn=lambda _: False, gs_equiv_mapper=gs_equiv_mapper, decomp_context=context
        )
        widgetizer.widgetize(circuit)
    qms = [args[1]["context"].qubit_manager for args in mock_decomp.call_args_list]
    assert qms[0] != qms[1]
    assert all([isinstance(qm, cirq.SimpleQubitManager) for qm in qms])


def test_widgetization_uses_passed_qubit_manager(get_circuit_equiv_map_small_fn):
    """Context used by decompose_once should employ the same qubit manager for all decompositions."""
    circuit, gs_equiv_mapper, small_enough_fn = get_circuit_equiv_map_small_fn
    qb_manager = cirq.SimpleQubitManager("simple_qm")
    context = cirq.DecompositionContext(qubit_manager=qb_manager)
    with patch("cirq.protocols.decompose_protocol.decompose_once") as mock_decomp:
        widgetizer = widgetization.CirqWidgetizer(
            small_enough_fn=small_enough_fn, gs_equiv_mapper=gs_equiv_mapper, decomp_context=context
        )
        widgetizer.widgetize(circuit)
    qms = [args[1]["context"].qubit_manager for args in mock_decomp.call_args_list]
    assert all([qm == qb_manager for qm in qms])


def test_CirqCircuitGraphNode_returns_correct_id():
    """When creating a CirqCircuitGraphNode, the id should be calculated correctly using passed equiv_id_fn."""

    def equiv_id_fn(circuit):
        return ("foo", "bar")

    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    node = widgetization.CirqCircuitGraphNode(circuit, equiv_id_fn)
    assert node.node_id == abs(hash(("foo", "bar")))


def test_CirqCircuitGraphNode_returns_correct_id_for_default_equiv_fn():
    """When creating a CirqCircuitGraphNode, the id should be calculated correctly using default equiv_id_fn."""
    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    node = CirqCircuitGraphNode(circuit)
    equiv_fn_result = ((cirq.X, (cirq.LineQubit(0),)),)  # should be output of default equiv_id_fn
    should_be = abs(hash(equiv_fn_result))
    actual = node.node_id
    assert actual == should_be


def test_widgetization_leaves_property_is_correct_given_graph(example_graph):
    """WidgetizationResult leaves attribute should be correct given a decomposition graph."""
    graph, leaf_nodes, _ = example_graph
    result = widgetization.WidgetizationResult(graph)

    should_be = set(leaf_nodes)
    actual = set(result.leaves)

    assert actual == should_be


def test_widgetization_root_node_property_is_correct_given_graph(example_graph):
    """WidgetizationResult root_node attribute should be correct given a decomposition graph."""
    graph, _, root_node = example_graph
    result = widgetization.WidgetizationResult(graph)

    should_be = root_node
    actual = result.root_node

    assert actual == should_be


def test_widgetization_ordered_leaf_nodes_property_is_correct_given_graph(example_graph):
    """WidgetizationResult root_node attribute should be correct given a decomposition graph."""
    graph, _, root_node = example_graph
    result = widgetization.WidgetizationResult(graph)

    should_be = (
        [(111, 5), (222, 7), (333, 11)]
        + ([(111, 5), (222, 7), (333, 11)] * 13 + [(111, 17)]) * 2
        + [(111, 5), (222, 7), (333, 11)] * 3
    )
    actual = result.ordered_leaf_node_ids
    assert actual == should_be


def test_widgetization_creates_correct_graph(example_circuit, example_graph):
    """Widgetization should create the correct graph."""
    # decompose if tags are larger than 3
    def small_enough_fn(circ):
        return sum([op.tags[0] for op in circ.all_operations()]) <= 3

    # circuits with same tag are "graph state equivalent"
    def equiv_id_fn(op):
        return op.tags[0]

    # graph state equivalence function (circuits with same tag are equivalent)
    gs_equiv_mapper = gseq.GraphStateEquivalenceMapper(op_labellers={None: equiv_id_fn})

    # example circuit
    circuit = example_circuit

    # Create graph
    widgetizer = widgetization.CirqWidgetizer(small_enough_fn=small_enough_fn, gs_equiv_mapper=gs_equiv_mapper)
    result = widgetizer.widgetize(circuit)
    actual = result.graph

    should_be, _, _ = example_graph  # the graph created by this circuit should be identical to the example graph
    assert nx.is_isomorphic(actual, should_be)


def test_widgetization_circuit_property_is_correct_given_graph(example_circuit):
    """WidgetizationResult root_node attribute should be correct given a decomposition graph."""

    def small_enough_fn(circ):
        return sum([op.tags[0] for op in circ.all_operations()]) <= 3

    # circuits with same tag are "graph state equivalent"
    def equiv_id_fn(op):
        return op.tags[0]

    # graph state equivalence function (circuits with same tag are equivalent)
    gs_equiv_mapper = gseq.GraphStateEquivalenceMapper(op_labellers={None: equiv_id_fn})

    # example circuit
    circuit = example_circuit

    # Create graph
    widgetizer = widgetization.CirqWidgetizer(small_enough_fn=small_enough_fn, gs_equiv_mapper=gs_equiv_mapper)
    result = widgetizer.widgetize(circuit)

    should_be = cirq.decompose(example_circuit)
    actual = cirq.decompose(result.circuit)
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 1), (10, 19), (7, 19), (3, 19)])
def test_correct_graph_size_without_prefab(maxq, expected_size, circuit):
    """Should return the correct number of nodes for the graph of the circuit. Uses decomp def, not pre fab defs.

    Maxq = 20 -> 1 node
    maxq = 10 -> 19 nodes (root node, 9 CNOT(a,b), 9 CNOT(b,a))
    maxq = 7 -> 19 nodes (root node, 9 CNOT(a,b), 9 CNOT(b,a))
    maxq = 3 -> 19 nodes (root node, 9 CNOT(a,b), 9 CNOT(b,a))
    """
    graph = get_widgetized_graph(circuit, maxq)
    should_be = expected_size
    actual = len(graph)
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 1), (10, 5), (7, 5), (3, 12)])
def test_correct_graph_size_with_prefab(maxq, expected_size, prefab_dict1, circuit):
    """Should return the correct number of nodes for the graph of the circuit. Uses pre fab1 defs.

    Maxq = 20 -> 1 node
    maxq = 10 -> 5 nodes (root node, 4 from prefab1 def of MyGate)
    maxq = 7 -> 5 nodes (root node, 4 from prefab1 def of MyGate)
    maxq = 3 -> 12 nodes (root, 4 from prefab1 def of MyGate, 3 unique CNOT(a,b), 4 CNOT (b,a) in MySmallerGate decomp)
    """
    graph = get_widgetized_graph(circuit, maxq, prefab_dict1)
    should_be = expected_size
    actual = len(graph)
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 1), (10, 5), (7, 5), (3, 7)])
def test_correct_graph_size_with_prefab2(maxq, expected_size, prefab_dict2, circuit):
    """Should return the correct number of nodes for the graph of the circuit. Uses pre fab2 defs.

    Maxq = 20 -> 1 node
    maxq = 10 -> 5 nodes (root node, 4 from prefab2 def of MyGate)
    maxq = 7 -> 5 nodes (root node, 4 from prefab2 def of MyGate)
    maxq = 3 -> 7 nodes (root, 4 from prefab2 def of MyGate, 2 from prefab2 def of MySmallerGate)
    """
    graph = get_widgetized_graph(circuit, maxq, prefab_dict2)
    should_be = expected_size
    actual = len(graph)
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 0), (10, 27), (7, 27), (3, 27)])
def test_correct_number_edges_without_prefab(maxq, expected_size, circuit):
    """Should return the correct number of edges for the graph of the circuit. Uses decomp defs.

    Maxq = 20 -> 0 edges
    maxq = 10 -> 27 edges (27 edges from root node to CNOTs from decomp def)
    maxq = 7 -> 27 edges (27 edges from root node to CNOTs from decomp def)
    maxq = 3 -> 27 edges (27 edges from root node to CNOTs from decomp def)
    """
    graph = get_widgetized_graph(circuit, maxq)
    should_be = expected_size
    actual = len(graph.edges)
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 0), (10, 4), (7, 4), (3, 16)])
def test_correct_number_edges_with_prefab(maxq, expected_size, circuit, prefab_dict1):
    """Should return the correct number of edges for the graph of the circuit. Uses prefab1 defs.

    Maxq = 20 -> 0 edges
    maxq = 10 -> 4 edges (4 edges from root node to gates in prefab1 def)
    maxq = 7 -> 4 edges (4 edges from root node to gates in prefab1 def)
    maxq = 3 -> 12 edges (4 edges from root node to gates in prefab1 def, 12 edges from MySmallerGate to unique CNOTS)
    """
    graph = get_widgetized_graph(circuit, maxq, prefab_dict1)
    should_be = expected_size
    actual = len(graph.edges)
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 0), (10, 4), (7, 4), (3, 6)])
def test_correct_number_edges_with_prefab2(maxq, expected_size, circuit, prefab_dict2):
    """Should return the correct number of edges for the graph of the circuit. Uses prefab2 defs.

    Maxq = 20 -> 0 edges
    maxq = 10 -> 4 edges (4 edges from root node to gates in prefab2 def)
    maxq = 7 -> 4 edges (4 edges from root node to gates in prefab2 def)
    maxq = 3 -> 6 edges (4 edges from root to gates in prefab2 def, 2 edges from MySmallerGate to gates in prefab2)
    """
    graph = get_widgetized_graph(circuit, maxq, prefab_dict2)
    should_be = expected_size
    actual = len(graph.edges)
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 0), (10, 27), (7, 27), (3, 27)])
def test_correct_edge_weights_without_prefab(maxq, expected_size, circuit):
    """Should return the correct total weight of edges for the graph of the circuit. Uses decomp defs.

    Maxq = 20 -> 0 weight (0 edges)
    maxq = 10 -> 27 weight (weight 1 for 27 edges from root node to CNOT gates in decomp)
    maxq = 7 -> 27 weight (weight 1 for 27 edges from root node to CNOT gates in decomp)
    maxq = 3 -> 27 weight (weight 1 for 27 edges from root node to CNOT gates in decomp)
    """
    graph = get_widgetized_graph(circuit, maxq)
    should_be = expected_size
    actual = graph.size(weight="weight")
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 0), (10, 25), (7, 25), (3, 37)])
def test_correct_edge_weights_with_prefab(maxq, expected_size, circuit, prefab_dict1):
    """Should return the correct total weight of edges for the graph of the circuit. Uses prefab1 defs.

    Maxq = 20 -> 0 weight (0 edges)
    maxq = 10 -> 25 weight (5 + 10 + 7 + 3, according to prefab1)
    maxq = 10 -> 25 weight (5 + 10 + 7 + 3, according to prefab1)
    maxq = 10 -> 37 weight (5 + 10 + 7 + 3, according to prefab1 + 12 weight 1 edges to CNOTS)
    """
    graph = get_widgetized_graph(circuit, maxq, prefab_dict1)
    should_be = expected_size
    actual = graph.size(weight="weight")
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 0), (10, 25), (7, 25), (3, 38)])
def test_correct_edge_weights_with_prefab2(maxq, expected_size, circuit, prefab_dict2):
    """Should return the correct total weight of edges for the graph of the circuit. Uses prefab2 defs.

    Maxq = 20 -> 0 weight (0 edges)
    maxq = 10 -> 25 weight (5 + 10 + 7 + 3, according to prefab2)
    maxq = 10 -> 25 weight (5 + 10 + 7 + 3, according to prefab2)
    maxq = 10 -> 38 weight (5 + 10 + 7 + 3, according to prefab2 + 2 + 11 edges from MySmallerGate)
    """
    graph = get_widgetized_graph(circuit, maxq, prefab_dict2)
    should_be = expected_size
    actual = graph.size(weight="weight")
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 1), (10, 1), (7, 1), (3, 1)])
def test_correct_graph_size_all_gates_equiv(maxq, expected_size, circuit):
    """Should return only a single node since all operations are defined to be gs equiv."""
    gs_equiv_mapper = gseq.GraphStateEquivalenceMapper(use_gate_name=False, use_qubits=False, op_labellers={})
    graph = get_widgetized_graph(circuit, maxq, gs_equiv_mapper=gs_equiv_mapper)
    should_be = expected_size
    actual = len(graph)
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 1), (10, 2), (7, 2), (3, 2)])
def test_correct_graph_size_gate_names_not_equiv(maxq, expected_size, circuit):
    """Should return 2 nodes since all operations with the same gate name are defined to be gs equiv.

    Note: due to decomp def, MyGate consists of all CNOT gates, which are all treated as equivalent.
    """
    gs_equiv_mapper = gseq.GraphStateEquivalenceMapper(use_gate_name=True, use_qubits=False, op_labellers={})
    graph = get_widgetized_graph(circuit, maxq, gs_equiv_mapper=gs_equiv_mapper)
    should_be = expected_size
    actual = len(graph)
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 1), (10, 1), (7, 1), (3, 1)])
def test_correct_graph_size_all_gates_equiv_prefab(maxq, expected_size, circuit, prefab_dict1):
    """Should return only a single node since all operations are defined to be gs equiv."""
    gs_equiv_mapper = gseq.GraphStateEquivalenceMapper(use_gate_name=False, use_qubits=False, op_labellers={})
    graph = get_widgetized_graph(circuit, maxq, gs_equiv_mapper=gs_equiv_mapper, pre_fab_dict=prefab_dict1)
    should_be = expected_size
    actual = len(graph)
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 1), (10, 5), (7, 5), (3, 5)])
def test_correct_graph_size_gate_names_not_equiv_prefab(maxq, expected_size, circuit, prefab_dict1):
    """Should return 5 nodes since all operations with same gate name are defined to be gs equiv.

    Note: due to prefab1, MyGate consists of 4 unique gates.
    """
    gs_equiv_mapper = gseq.GraphStateEquivalenceMapper(use_gate_name=True, use_qubits=False, op_labellers={})
    graph = get_widgetized_graph(circuit, maxq, gs_equiv_mapper=gs_equiv_mapper, pre_fab_dict=prefab_dict1)
    should_be = expected_size
    actual = len(graph)
    assert actual == should_be


@pytest.mark.parametrize("maxq,expected_size", [(20, 1), (10, 5), (7, 5), (3, 7)])
def test_correct_graph_size_gate_names_not_equiv_prefab2(maxq, expected_size, circuit, prefab_dict2):
    """Should return 7 nodes since all operations with same gate name are defined to be gs equiv.

    Note: due to prefab2, MyGate consists of 4 unique gates, and MySmallerGate consists of 2 unique gates.
    """
    gs_equiv_mapper = gseq.GraphStateEquivalenceMapper(use_gate_name=True, use_qubits=False, op_labellers={})
    graph = get_widgetized_graph(circuit, maxq, gs_equiv_mapper=gs_equiv_mapper, pre_fab_dict=prefab_dict2)
    should_be = expected_size
    actual = len(graph)
    assert actual == should_be


def test_widgets_and_stitches_outputs_correctly(example_graph):
    """widgets_and_stitches should return the correct dictionary given a graph."""
    graph, _, root_node = example_graph
    result = widgetization.WidgetizationResult(graph)

    should_be_widgets = {"W-0": 184, "W-1": 210, "W-2": 330}  # node_id 111,  # node_id 222,  # node_id 333

    # arrived at by analyzing the example graph, repetitions, and edge weights
    should_be_stitches = {
        ("W-0", "W-0"): (5 - 1) + ((5 - 1) * 13 + (17 - 1)) * 2 + (5 - 1) * 3 + 2,
        ("W-0", "W-1"): 1 + 13 * 2 + 3,
        ("W-1", "W-1"): (7 - 1) + ((7 - 1) * 13) * 2 + (7 - 1) * 3,
        ("W-1", "W-2"): 1 + 13 * 2 + 3,
        ("W-2", "W-2"): (11 - 1) + ((11 - 1) * 13) * 2 + (11 - 1) * 3,
        ("W-2", "W-0"): 1 + 13 * 2 + 2,
    }
    actual_widgets = result.widgets_and_stitches["widgets"]
    actual_stitches = result.widgets_and_stitches["stitches"]
    assert actual_widgets == should_be_widgets
    assert actual_stitches == should_be_stitches


def test_widgetization_creates_correct_widgets_and_stitches_from_multimoment_circuit(qbs, prefab_dict3):
    """Widgetization should create the graph, widgets, and stitches when a subcircuit has multiple moments."""

    def tags(op):
        # Operations that naturally decompose in cirq will not have tags, so we just give a default value
        # This doesn't change the test, but avoids index errors.
        if len(op.tags) == 1:
            return op.tags[0]
        else:
            return 1

    # decompose if tags are larger than 3
    def small_enough_fn(circ):
        return sum([tags(op) for op in circ.all_operations()]) <= 3

    # circuits with same tag are "graph state equivalent"
    def equiv_id_fn(op):
        return tags(op)

    # graph state equivalence function (circuits with same tag are equivalent)
    gs_equiv_mapper = gseq.GraphStateEquivalenceMapper(op_labellers={None: equiv_id_fn})

    # circuit
    # tags are used to determine if an operation should be decomposed. Any operation with a tag larger than 3 will
    # be decomposed
    circuit = cirq.Circuit(MyGate().on(*qbs).with_tags(47))

    # Create graph
    widgetizer = widgetization.CirqWidgetizer(
        small_enough_fn=small_enough_fn, gs_equiv_mapper=gs_equiv_mapper, pre_fab_dict=prefab_dict3
    )
    result = widgetizer.widgetize(circuit)
    should_be_w_and_s = {
        "widgets": {"W-0": 7, "W-1": 7, "W-2": 14, "W-3": 77, "W-4": 7},
        "stitches": {
            ("W-0", "W-1"): 7,
            ("W-2", "W-2"): 7,
            ("W-1", "W-2"): 7,
            ("W-3", "W-3"): 70,
            ("W-2", "W-3"): 7,
            ("W-3", "W-4"): 7,
            ("W-4", "W-0"): 6,
        },
    }
    should_be_num_nodes = 9
    should_be_num_edges = 8
    should_be_num_widgets = 5

    actual_w_and_s = result.widgets_and_stitches
    actual_num_nodes = len(result.graph)
    actual_num_edges = len(result.graph.edges)
    actual_num_widgets = len(result.widgets)

    assert actual_w_and_s == should_be_w_and_s
    assert actual_num_nodes == should_be_num_nodes
    assert actual_num_edges == should_be_num_edges
    assert actual_num_widgets == should_be_num_widgets
