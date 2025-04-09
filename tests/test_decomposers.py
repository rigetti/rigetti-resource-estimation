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

"""Unit tests for `rigetti_resource_estimation` decomposers module."""


from rigetti_resource_estimation import decomposers, translators, transpile
from qualtran.bloqs.basic_gates import TGate, CNOT, XGate, YGate, ZGate, SGate
from qualtran.cirq_interop._bloq_to_cirq import BloqAsCirqGate
from rigetti_resource_estimation.decomposers import OperationDecomposer, CirqDecomposer
from qualtran.bloqs.mod_arithmetic import ModAddK
from qualtran import Bloq, Register, QBit, Signature
from qualtran.bloqs.basic_gates.rotation import ZPowGate as qt_zpowgate
import cirq
import cirq.testing as ct
import pytest
from unittest.mock import patch
import numpy as np
import qualtran as qt

default_transpiler = transpile.CirqTranspiler(translators.DEFAULT_TRANSLATORS)


def test_bloq_as_cirq_gate_handles_basic_gate_bloqs():
    """Should allow decomposition of cirq wrapped basic bloqs without raising an exception or infinite recursion."""
    bloqs_to_handle = [TGate(), CNOT(), XGate(), YGate(), ZGate(), SGate()]
    circuit = cirq.Circuit()
    for bloq in bloqs_to_handle:
        n_qubits = bloq.signature.n_qubits()
        op = BloqAsCirqGate(bloq).on(*cirq.LineQubit.range(n_qubits))
        circuit.append(op)
    default_transpiler = transpile.CirqTranspiler(
        translators.CIRQ2CABALISER_SIMPLE_TRANSLATORS + translators.CIRQ2CABALISER_OTHER_TRANSLATORS
    )
    circuit_decomposer = decomposers.CirqDecomposer(decomposers.DEFAULT_DECOMPOSERS)
    circuit_decomposer.decompose_and_qb_map(circuit=circuit, keep=default_transpiler.is_compatible)


def test_bloq_as_cirq_gate_raises_value_error():
    """Should raise ValueError if decomposer tries to decompose cirq wrapped bloq without knowing how to."""

    # arbitrary bloq, without decomposition instructions
    class TestBloq(Bloq):
        @property
        def signature(self):
            return Signature([Register("qb1", QBit())])

    circuit = cirq.Circuit()
    op = BloqAsCirqGate(TestBloq()).on(*cirq.LineQubit.range(1))
    circuit.append(op)
    default_transpiler = transpile.CirqTranspiler(
        translators.CIRQ2CABALISER_SIMPLE_TRANSLATORS + translators.CIRQ2CABALISER_OTHER_TRANSLATORS
    )
    circuit_decomposer = decomposers.CirqDecomposer(decomposers.DEFAULT_DECOMPOSERS)
    with pytest.raises(ValueError):
        circuit_decomposer.decompose_and_qb_map(circuit=circuit, keep=default_transpiler.is_compatible)


def test_qualtran_z_power_gate_decomposes_without_max_recursion():
    """Should decompose properly without encountering max recursion using default decomposers and translators."""
    gate = qt_zpowgate(exponent=0.2)  # exponent is arbitrary
    circuit = cirq.Circuit()
    circuit.append(gate.on(cirq.LineQubit(0)))
    default_transpiler = transpile.CirqTranspiler(translators.DEFAULT_TRANSLATORS)
    circuit_decomposer = decomposers.CirqDecomposer(decomposers.DEFAULT_DECOMPOSERS)
    decomposed_circ, qb_map = circuit_decomposer.decompose_and_qb_map(
        circuit=circuit, keep=default_transpiler.is_compatible
    )


def test_decompose_and_qbmap_uses_default_qubit_manager():
    """Context used by decompose_once should employ the same cirq.SimpleQubitManager for each decomposition."""
    circuit = ct.random_circuit(qubits=20, n_moments=100, op_density=0.7)
    with patch("cirq.protocols.decompose_protocol.decompose_once", wraps=cirq.decompose_once) as mock_decomp:
        circuit_decomposer = decomposers.CirqDecomposer(decomposers.DEFAULT_DECOMPOSERS)
        circuit_decomposer.decompose_and_qb_map(circuit=circuit, keep=default_transpiler.is_compatible)
    qms = [args[1]["context"].qubit_manager for args in mock_decomp.call_args_list]
    assert all([qm == qms[0] for qm in qms])
    assert all([isinstance(qm, cirq.SimpleQubitManager) for qm in qms])


def test_decomposeand_qbmap_uses_passed_qubit_manager():
    """Context used by decompose_once should employ the same qubit manager for all decompositions."""
    circuit = ct.random_circuit(qubits=20, n_moments=100, op_density=0.7)
    qb_manager = cirq.SimpleQubitManager("simple_qm")
    context = cirq.DecompositionContext(qubit_manager=qb_manager)
    with patch("cirq.protocols.decompose_protocol.decompose_once", wraps=cirq.decompose_once) as mock_decomp:
        circuit_decomposer = decomposers.CirqDecomposer(decomposers.DEFAULT_DECOMPOSERS)
        circuit_decomposer.decompose_and_qb_map(circuit=circuit, keep=default_transpiler.is_compatible, context=context)
    qms = [args[1]["context"].qubit_manager for args in mock_decomp.call_args_list]
    assert all([qm == qb_manager for qm in qms])


@pytest.fixture
def z_pow_exponent_to_gate_pairs():
    """Pairs of rotation angle (in multiples of pi) to cirq Gate.

    For example, (1,cirq.Z) indicates that rotation angles of 1 * pi + 2 * pi * k (k an integer) is a cirq.Z gate.
    """
    return [(0, cirq.I), (1, cirq.Z), (1 / 4, cirq.T), (1 / 2, cirq.S), (3 / 2, cirq.S**-1)]


@pytest.fixture
def op_decomposer_params():
    """Example sets of parameters for constructing an operation decomposer."""
    params1 = {
        "framework": "cirq",
        "decomposer_type": "intercepting",
        "labels": ["label1", "label2"],
        "meta": {"Meta1": 0.1},
        "decomposer_id": "op_decomp1",
        "can_apply_to_fn": lambda op: op.gate == cirq.Z,
        "decompose_fn": lambda op: (cirq.S**-1).on(op.qubits[0]),
    }

    params2 = {
        "framework": "cirq",
        "decomposer_type": "fallback",
        "labels": ["label1", "label2"],
        "meta": {"Meta1": 0.1},
        "decomposer_id": "op_decomp1",
        "can_apply_to_fn": lambda op: op.gate == cirq.Y,
        "decompose_fn": lambda op: cirq.S.on(op.qubits[0]),
    }

    return [params1, params2]


@pytest.fixture
def op_decomposers(op_decomposer_params):
    """Create an example set of operation decomposers."""
    return [OperationDecomposer(**params) for params in op_decomposer_params]


@pytest.fixture
def qubits():
    """Create an example set of qubits for use in testing."""
    qb1 = cirq.NamedQubit("Bob")
    qb2 = cirq.NamedQubit("Alice")
    qb3 = cirq.NamedQubit("Gord")
    qb4 = cirq.NamedQubit("Neil")
    return [qb1, qb2, qb3, qb4]


@pytest.fixture
def circuits(qubits):
    """Make a simple example circuits for testing."""
    qbs = qubits
    circuit = cirq.Circuit()
    circuit.append(cirq.S(qbs[0]))
    circuit.append(cirq.X(qbs[1]))
    circuit.append(cirq.Y(qbs[2]))
    circuit.append(cirq.CZ(qbs[1], qbs[2]))
    circuit.append(cirq.Z(qbs[3]))

    decomp_ops = [cirq.S(qbs[0]), cirq.X(qbs[1]), cirq.S(qbs[2]), cirq.CZ(qbs[1], qbs[2]), (cirq.S**-1)(qbs[3])]
    only_decomp_y_ops = decomp_ops.copy()
    only_decomp_y_ops[-1] = cirq.Z(qbs[3])  # Z doesn't get 'decomposed'

    return circuit, cirq.Circuit(decomp_ops), cirq.Circuit(only_decomp_y_ops)


def test_operation_decomposer_attributes(op_decomposer_params):
    """Should have the correct attributes after construction."""
    decomp_params1, decomp_params2 = op_decomposer_params
    decomposer1 = OperationDecomposer(**decomp_params1)
    decomposer2 = OperationDecomposer(**decomp_params2)
    for attribute in ["framework", "decomposer_type", "decomposer_id", "labels", "meta"]:
        assert getattr(decomposer1, attribute) == decomp_params1[attribute]
        assert getattr(decomposer2, attribute) == decomp_params2[attribute]


def test_operation_decomposer_convert(op_decomposer_params):
    """Example decomposers should decompose operations properly."""
    decomposer1 = OperationDecomposer(**op_decomposer_params[0])
    decomposer2 = OperationDecomposer(**op_decomposer_params[1])

    actual_ops = []
    should_be_ops = []
    actual_ops.append(decomposer1.convert(cirq.Z(cirq.LineQubit(0))))
    should_be_ops.append((cirq.S**-1)(cirq.LineQubit(0)))
    actual_ops.append(decomposer2.convert(cirq.Y(cirq.LineQubit(1))))
    should_be_ops.append(cirq.S(cirq.LineQubit(1)))

    for actual, should_be in zip(actual_ops, should_be_ops):
        assert actual == should_be


def test_decomposer_counts_correctly(op_decomposers):
    """Decomposers should track the number of times they are called correctly."""
    decomposer1, decomposer2 = op_decomposers

    should_be_decomp1_calls = 5  # arbitrary
    should_be_decomp2_calls = 7  # arbitrary

    for _ in range(should_be_decomp1_calls):
        decomposer1.convert(cirq.Z(cirq.LineQubit(0)))

    for _ in range(should_be_decomp2_calls):
        decomposer2.convert(cirq.Y(cirq.LineQubit(1)))

    actual_decomp1_calls = decomposer1.times_called
    assert actual_decomp1_calls == should_be_decomp1_calls

    actual_decomp2_calls = decomposer2.times_called
    assert actual_decomp2_calls == should_be_decomp2_calls


def test_cirq_decomposer_raises_if_not_all_cirq_framework(op_decomposer_params):
    """Should raise if not all operation decomposers are in the cirq framework."""
    params1, params2 = op_decomposer_params
    params3 = params1.copy()
    params3["framework"] = "qiskit"

    op_decomposers = [OperationDecomposer(**params) for params in [params1, params2, params3]]
    with pytest.raises(ValueError):
        CirqDecomposer(op_decomposers)


def test_op_decomposers_dict_attribute_is_correct(op_decomposers):
    """Operation decomposers should have correct attributes."""
    decomposer1, decomposer2 = op_decomposers
    decomposer3 = decomposer1
    cirq_decomposer = CirqDecomposer(op_decomposers + [decomposer3])
    assert cirq_decomposer.op_decomposers["intercepting"] == [decomposer1, decomposer3]
    assert cirq_decomposer.op_decomposers["fallback"] == [decomposer2]


def test_qb_map_is_correct(op_decomposers, circuits, qubits):
    """Operation decomposers should return the correct qb map."""
    qbs = qubits
    cirq_decomposer = CirqDecomposer(op_decomposers)
    circuit, should_be_decomposed_keep_y, should_be_decomposed = circuits

    def dont_keep_y(op):
        return op.gate != cirq.Y

    def only_decomp_y(op):
        return op.gate != cirq.Y  #  returns False if Y gate

    _, actual_qb_map = cirq_decomposer.decompose_and_qb_map(circuit, keep=None)
    assert set(qbs) == set(actual_qb_map.keys())
    _, actual_qb_map = cirq_decomposer.decompose_and_qb_map(circuit, keep=dont_keep_y)
    assert set(qbs) == set(actual_qb_map.keys())


def test_decomposed_is_correct(op_decomposers, circuits):
    """CirqDecomposer, when created with set of op_decomposers, should decompose circuit properly."""
    cirq_decomposer = CirqDecomposer(op_decomposers)
    circuit, should_be_decomposed, should_be_decomposed_only_y = circuits

    def dont_keep_y(op):
        return op.gate != cirq.Y

    # decompose without a keep function (decompose down to cirq default gate set)
    actual_decomposed, _ = cirq_decomposer.decompose_and_qb_map(circuit, keep=None)
    assert actual_decomposed == should_be_decomposed

    # decompose only y
    actual_decomposed, _ = cirq_decomposer.decompose_and_qb_map(circuit, keep=dont_keep_y)
    assert actual_decomposed == should_be_decomposed_only_y


def test_decomposed_is_unitarily_correct_for_default_decomposers_and_transpiler():
    """Using default decomposers and translators, ensure random circuit is decomposed to same unitary."""
    circuit_decomposer = decomposers.CirqDecomposer(decomposers.DEFAULT_DECOMPOSERS)
    default_transpiler = transpile.CirqTranspiler(translators.DEFAULT_TRANSLATORS)

    for _ in range(100):
        rc = cirq.testing.random_circuit(qubits=10, n_moments=5, op_density=0.7, random_state=42)
        decomposed, _ = circuit_decomposer.decompose_and_qb_map(rc, keep=default_transpiler.is_compatible)
        cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(rc), cirq.unitary(decomposed), atol=1e-8)


def test_modk_decomposes_with_default_decomposers():
    """Circuit with ModAddK gate should decompose without issue."""
    modk = ModAddK(bitsize=1, mod=2, add_val=1, cvs=(0, 0)).on(*cirq.LineQubit.range(3))
    circuit = cirq.Circuit()
    circuit.append(modk)
    default_transpiler = transpile.CirqTranspiler(
        translators.CIRQ2CABALISER_SIMPLE_TRANSLATORS + translators.CIRQ2CABALISER_OTHER_TRANSLATORS
    )
    circuit_decomposer = decomposers.CirqDecomposer(decomposers.DEFAULT_DECOMPOSERS)
    circuit_decomposer.decompose_and_qb_map(circuit=circuit, keep=default_transpiler.is_compatible)


@pytest.mark.parametrize("tolerance", [(1e-6), (1e-8), (1e-10)])
@pytest.mark.parametrize("pi_multiple", [(i) for i in 2 * np.arange(-2, 2)])
@pytest.mark.parametrize("tol_multiple", [(-0.3), (0.8), (1.2), (-2.1)])
def test_process_z_rot(z_pow_exponent_to_gate_pairs, tol_multiple, pi_multiple, tolerance):
    """Processing of z rotation gates should be handled correctly."""
    qb = cirq.LineQubit(0)
    for exponent, gate in z_pow_exponent_to_gate_pairs:
        exponent = exponent + pi_multiple + tol_multiple * tolerance
        op = cirq.ZPowGate(exponent=exponent).on(qb)
        if abs(tol_multiple) <= 1:
            # within tolerance
            should_be = gate.on(qb)
        else:
            # outside of tolerance
            should_be = op

        actual = decomposers.process_z_rot(op, tolerance)
        assert actual == should_be


@pytest.mark.parametrize("exponent", [(0.3), (-0.2), (1.1), (1), (2), (1 / 2), (0.01)])
def test_ry_to_rz_decomp(exponent):
    """Decomposition of y rotations should be unitarily correct."""
    op = (cirq.YPowGate(exponent=exponent)).on(cirq.LineQubit(0))
    circuit = cirq.Circuit(op)
    new_circuit = cirq.Circuit(decomposers.ry_to_rz_decomp(op))

    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(circuit), cirq.unitary(new_circuit), atol=1e-8)


@pytest.mark.parametrize("exponent", [(0.3), (-0.2), (1.1), (1), (2), (1 / 2), (0.01)])
def test_rx_to_rz_decomp(exponent):
    """Decomposition of x rotations should be unitarily correct."""
    op = (cirq.XPowGate(exponent=exponent)).on(cirq.LineQubit(0))
    circuit = cirq.Circuit(op)
    new_circuit = cirq.Circuit(decomposers.rx_to_rz_decomp(op))

    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(circuit), cirq.unitary(new_circuit), atol=1e-8)


def test_qualtran_bloq_as_cirq_fn():
    """Decomposition of qualtran CNOT bloq as a cirq gate should be correct."""
    qubits = cirq.LineQubit.range(2)
    bloq = qt.bloqs.basic_gates.CNOT()
    gate = qt.cirq_interop.BloqAsCirqGate(bloq=bloq)
    op = gate.on(*qubits)
    actual = decomposers.qualtran_bloq_as_cirq(op)
    should_be = cirq.CNOT.on(*qubits)

    assert actual == should_be


@pytest.mark.parametrize("exponent", [(0.3), (-0.2), (1.1), (1), (2), (1 / 2), (0.01)])
@pytest.mark.parametrize("phase_shift", [(0.1), (-0.3), (1.01), (1), (3), (1 / 2), (0.05)])
def test_czpower_decomp(exponent, phase_shift):
    """Decomposition of CZPowGate should be unitarily correct."""
    op = (cirq.CZPowGate(exponent=exponent, global_shift=phase_shift)).on(*cirq.LineQubit.range(2))
    circuit = cirq.Circuit(op)
    new_circuit = cirq.Circuit(decomposers.czpower_decomp(op))

    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(circuit), cirq.unitary(new_circuit), atol=1e-8)
