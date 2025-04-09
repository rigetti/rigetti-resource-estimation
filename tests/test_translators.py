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

"""Test for `rigetti_resource_estimation` translators module."""
import pytest
import cirq
from rigetti_resource_estimation.translators import OperationTranslator, CirqToCabaliser
from rigetti_resource_estimation import translators
from cabaliser import gates as cab_gates
import numpy as np

TRANSLATORS = translators.DEFAULT_TRANSLATORS


@pytest.fixture
def op_translator_params():
    """Example sets of parameters for constructing an operation translator."""
    params1 = {
        "from_framework": "cirq",
        "to_framework": "cabaliser",
        "op_id": "translator_1",
        "labels": ["label1", "label2"],
        "meta": {"Meta1": 0.1},
        "can_apply_to_fn": lambda op: op.gate == cirq.Z,
        "translate_fn": lambda op, qbmap: (1001, tuple(qbmap[qb] for qb in op.qubits)),
    }
    params2 = {
        "from_framework": "cirq",
        "to_framework": "cabaliser",
        "op_id": "translator_2",
        "labels": ["label3", "label4"],
        "meta": {"Meta2": 0.2},
        "can_apply_to_fn": lambda op: op.gate == cirq.CZ,
        "translate_fn": lambda op, qbmap: (9999, tuple(qbmap[qb] for qb in op.qubits)),
    }

    return [params1, params2]


@pytest.fixture
def op_translators(op_translator_params):
    """Create an example set of operation translators."""
    return [OperationTranslator(**params) for params in op_translator_params]


@pytest.fixture
def qubits():
    """Create an example set of qubits for use in testing."""
    qb1 = cirq.NamedQubit("Bob")
    qb2 = cirq.NamedQubit("Alice")
    qb3 = cirq.NamedQubit("Gord")
    qb4 = cirq.NamedQubit("Neil")
    return [qb1, qb2, qb3, qb4]


@pytest.fixture
def circuit(qubits):
    """Make a simple example circuit for testing."""
    qbs = qubits
    circuit = cirq.Circuit()
    circuit.append(cirq.S(qbs[0]))
    circuit.append(cirq.X(qbs[1]))
    circuit.append(cirq.Y(qbs[2]))
    circuit.append(cirq.CZ(qbs[1], qbs[2]))
    circuit.append(cirq.Z(qbs[3]))

    return circuit


@pytest.fixture
def make_cirq_to_cabaliser_translators(op_translator_params):
    params1, params2 = op_translator_params
    params1.pop("to_framework")
    params2.pop("to_framework")
    params1.pop("from_framework")
    params2.pop("from_framework")
    translator1 = CirqToCabaliser(**params1)
    translator2 = CirqToCabaliser(**params2)
    return translator1, translator2


def test_operation_translator_attributes(op_translator_params):
    """Should have the correct attributes after construction."""
    translator_params1, translator_params2 = op_translator_params
    translator1 = OperationTranslator(**translator_params1)
    translator2 = OperationTranslator(**translator_params2)
    for attribute in ["to_framework", "from_framework", "op_id", "labels", "meta"]:
        assert getattr(translator1, attribute) == translator_params1[attribute]
        assert getattr(translator2, attribute) == translator_params2[attribute]


def test_operation_translator_convert(op_translator_params):
    """Example translators should translate operations properly."""
    translator1 = OperationTranslator(**op_translator_params[0])
    translator2 = OperationTranslator(**op_translator_params[1])
    qbmap1 = {cirq.LineQubit(0): 0}
    qbmap2 = {cirq.LineQubit(i): i for i in range(2)}

    actual_ops = []
    should_be_ops = []
    actual_ops.append(translator1.convert(cirq.Z(cirq.LineQubit(0)), qbmap1))
    should_be_ops.append((1001, (0,)))
    actual_ops.append(translator2.convert(cirq.CZ(*cirq.LineQubit.range(2)), qbmap2))
    should_be_ops.append((9999, (0, 1)))

    for actual, should_be in zip(actual_ops, should_be_ops):
        assert actual == should_be


def test_translator_counts_correctly(op_translators):
    """Translators should track the number of times they are called correctly."""
    translator1, translator2 = op_translators[:2]

    should_be_translator1_calls = 5  # arbitrary
    should_be_translator2_calls = 7  # arbitrary
    qbmap1 = {cirq.LineQubit(0): 0}
    qbmap2 = {cirq.LineQubit(i): i for i in range(2)}

    for _ in range(should_be_translator1_calls):
        translator1.convert(cirq.Z(cirq.LineQubit(0)), qbmap1)

    for _ in range(should_be_translator2_calls):
        translator2.convert(cirq.CZ(*cirq.LineQubit.range(2)), qbmap2)

    actual_decomp1_calls = translator1.times_called
    assert actual_decomp1_calls == should_be_translator1_calls

    actual_decomp2_calls = translator2.times_called
    assert actual_decomp2_calls == should_be_translator2_calls


def test_cirq_to_cabaliser_translator_attributes(make_cirq_to_cabaliser_translators):
    """CirqToCabaliser translators should have to and from framework set correctly."""
    for translator in make_cirq_to_cabaliser_translators:
        assert translator.from_framework == "cirq"
        assert translator.to_framework == "cabaliser"


@pytest.mark.parametrize("gate", [(gate) for gate in translators.DIRECT_CIRQ_TO_CABALISER_GATES])
def test_cirq_to_cabaliser_simple_translator(gate):
    """Simple translators should translate to cabaliser correctly."""
    cirq_gate = getattr(cirq, gate)
    translator = CirqToCabaliser(cirq_gate, labels=["clifford"], meta={"angle_tag": None})
    qbs = cirq.LineQubit.range(cirq_gate.num_qubits())
    op = cirq_gate.on(*qbs)
    qb_map = {cirq.LineQubit(i): i for i in range(len(qbs))}
    translated = translator.convert(op, qb_map)
    should_be_type = tuple
    should_be_int = getattr(cab_gates, gate)
    should_be_qbs = tuple(qb_map[qb] for qb in qbs)

    actual_type = type(translated)
    actual_int, actual_qbs = translated

    assert isinstance(translated, tuple)
    assert actual_type == should_be_type
    assert actual_int == should_be_int
    assert actual_qbs == should_be_qbs


@pytest.mark.parametrize("gate", [(gate) for gate in translators.SELF_INVERSE_GATES])
def test_self_inverse_gates_are_self_inverses(gate):
    """Self-inverse gates should actually be self inverses."""
    cirq_gate = getattr(cirq, gate)
    assert cirq_gate**-1 == cirq_gate


@pytest.mark.parametrize(
    "translator", [(translator) for translator in translators.CIRQ2CABALISER_SELF_INVERSE_TRANSLATORS]
)
def test_self_inverse_translators(translator):
    """Self-inverse translators should have the correct labels, and meta data."""
    should_be_labels = ["clifford"]
    should_be_meta = {"angle_tag": None}

    actual_labels = translator.labels
    actual_meta = translator.meta

    assert actual_labels == should_be_labels
    assert actual_meta == should_be_meta


@pytest.fixture
def self_inverse_gates():
    gates_to_check = translators.SELF_INVERSE_GATES
    exponents = [-1, -1.0]
    return [pow(getattr(cirq, gate), exponent) for gate in gates_to_check for exponent in exponents]


def test_self_inverse_gates_have_one_applicable_translator(self_inverse_gates):
    """Self-inverse translators should only be applicable to one type of operation."""
    for gate in self_inverse_gates:
        if gate.num_qubits() == 1:
            op = gate.on(cirq.LineQubit(0))
        if gate.num_qubits() == 2:
            op = gate.on(*cirq.LineQubit.range(2))
        applies_to = [translator.can_apply_to(op) for translator in TRANSLATORS]
        assert sum(applies_to) == 1


def test_self_inverse_gates_translate_as_non_inverse_version(self_inverse_gates):
    """Self-inverse operations should translate into the same operation as non-inverse operations do."""
    for gate in self_inverse_gates:
        if gate.num_qubits() == 1:
            op = gate.on(cirq.LineQubit(0))
            non_inv_op = cirq.inverse(gate).on(cirq.LineQubit(0))
        if gate.num_qubits() == 2:
            op = gate.on(*cirq.LineQubit.range(2))
            non_inv_op = cirq.inverse(gate).on(*cirq.LineQubit.range(2))
        for translator in TRANSLATORS:
            if translator.can_apply_to(op):
                actual = translator.convert(op, qb_map={qb: i for i, qb in enumerate(op.qubits)})
        for translator in translators.CIRQ2CABALISER_SIMPLE_TRANSLATORS:
            if translator.can_apply_to(non_inv_op):
                should_be = translator.convert(non_inv_op, qb_map={qb: i for i, qb in enumerate(op.qubits)})

        assert actual == should_be


def test_global_phase_translator():
    """Global phase translator should translate to cabaliser instruction correctly."""
    gate = cirq.GlobalPhaseGate(np.exp(4j))
    op = gate.on()
    qb_map = {}

    translator = translators.CIRQ2CABALISER_GLOBAL_PHASE
    assert translator.labels == ["global phase"]
    actual = translator.convert(op, qb_map)

    # Assume translated into identity on qubit 0
    should_be = (32, (0,))
    assert actual == should_be


def test_measurement_translator():
    """Measurement translator should translate to cabaliser instruction correctly."""
    gate = cirq.MeasurementGate(num_qubits=1)
    op = gate.on(cirq.LineQubit(0))
    qb_map = {qb: i for i, qb in enumerate(op.qubits)}

    translator = translators.CIRQ2CABALISER_MEASUREMENT
    assert translator.labels == ["measurement"]
    actual = translator.convert(op, qb_map)

    # Assume translated into identity on qubit 0
    should_be = (32, (0,))
    assert actual == should_be


def test_reset_translator():
    """Reset translator should translate to cabaliser instruction correctly."""
    gate = cirq.ResetChannel()
    op = gate.on(cirq.LineQubit(0))
    qb_map = {qb: i for i, qb in enumerate(op.qubits)}

    translator = translators.CIRQ2CABALISER_RESET
    assert translator.labels == ["reset"]
    actual = translator.convert(op, qb_map)

    # Assume translated into identity on qubit 0
    should_be = (32, (0,))
    assert actual == should_be


def test_T_translator():
    """T gate translator should translate to cabaliser correctly."""
    gate = cirq.T
    op = gate.on(cirq.LineQubit(0))
    qb_map = {qb: i for i, qb in enumerate(op.qubits)}

    translator = translators.CIRQ2CABALISER_T
    assert translator.meta == {"angle_tag": 31}
    assert translator.labels == ["t"]

    actual = translator.convert(op, qb_map)

    # T angle tag has been set at 31
    should_be = (cab_gates.RZ, (0, 31))
    assert actual == should_be


def test_invT_translator():
    """Inverse T operations should translate into correct cabaliser operation."""
    gate = cirq.T**-1
    op = gate.on(cirq.LineQubit(0))
    qb_map = {qb: i for i, qb in enumerate(op.qubits)}

    translator = translators.CIRQ2CABALISER_TDAG
    assert translator.meta == {"angle_tag": 333}
    assert translator.labels == ["t"]

    actual = translator.convert(op, qb_map)

    # Inverse T angle tag has been set at 333
    should_be = (cab_gates.RZ, (0, 333))
    assert actual == should_be


def test_rz_translator():
    """Translator for rz rotation with arbitrary angle should have correct meta, labels, and translate correctly."""
    gate = cirq.ZPowGate(exponent=0.3)
    op = gate.on(cirq.LineQubit(0))
    qb_map = {qb: i for i, qb in enumerate(op.qubits)}

    translator = translators.CIRQ2CABALISER_ARB_Z
    assert translator.meta == {"angle_tag": 77}
    assert translator.labels == ["rz"]

    actual = translator.convert(op, qb_map)

    # Inverse T angle tag has been set at 77
    should_be = (cab_gates.RZ, (0, 77))
    assert actual == should_be


@pytest.mark.parametrize("angle", [(0), (1 / 4), (1 / 2), (3 / 2), (7 / 4), (1)])
def test_rz_translator_does_not_apply_to_clifford_angles(angle):
    """Translator for arbitrary rz rotation should not apply to rotations with clifford angles."""
    gate = cirq.ZPowGate(exponent=angle)
    op = gate.on(cirq.LineQubit(0))

    translator = translators.CIRQ2CABALISER_ARB_Z
    assert not translator.can_apply_to(op)
