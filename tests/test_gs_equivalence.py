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

"""Test for `rigetti_resource_estimation` gs_equivalence module."""
import pytest
import cirq
from rigetti_resource_estimation import gs_equivalence as gseq
from pyLIQTR.qubitization.qubitized_gates import QubitizedRotation
from pyLIQTR.ProblemInstances.getInstance import getInstance
from pyLIQTR.circuits.operators.select_prepare_pauli import prepare_pauli_lcu
from pyLIQTR.BlockEncodings.getEncoding import getEncoding, VALID_ENCODINGS
from pyLIQTR.clam.lattice_definitions import SquareLattice
import numpy as np
import qualtran as qt


@pytest.fixture
def example_labellers():
    def label_x_y_z_h_gate(op):
        if op.gate == cirq.X or op.gate == cirq.Y:
            return ("foo", 3)
        if op.gate == cirq.Z or op.gate == cirq.H:
            return ("bar", 3)

    def label_s_T_gate(op):
        if op.gate == cirq.S or op.gate == cirq.T:
            return (2, False)

    return {
        cirq.X: label_x_y_z_h_gate,
        cirq.Y: label_x_y_z_h_gate,
        cirq.Z: label_x_y_z_h_gate,
        cirq.H: label_x_y_z_h_gate,
        cirq.S: label_s_T_gate,
        cirq.T: label_s_T_gate,
    }


@pytest.fixture
def example_circuit():
    qb0, qb1, qb2, qb3 = cirq.LineQubit.range(4)
    return cirq.Circuit([cirq.X(qb0), cirq.X(qb1), cirq.X(qb3)])


def test_gs_mapper_has_default_op_labellers():
    """Graph state mapper should come equipped with the default labellers."""
    mapper = gseq.GraphStateEquivalenceMapper()
    should_be = set(gseq.SUPPORTED_GATES.keys())
    actual = set(mapper.op_labellers.keys())
    assert actual == should_be


def test_gs_mapper_has_all_passed_op_labellers():
    """Graph state mapper should come equipped with all passed labellers."""
    new_labeller = {cirq.X: lambda op: (str(op), "foo", "bar")}
    mapper = gseq.GraphStateEquivalenceMapper(op_labellers=new_labeller)
    assert cirq.X in mapper.op_labellers.keys()


@pytest.mark.parametrize("use_gate,use_qubit", [(True, True), (True, False), (False, True), (False, False)])
def test_gs_mapper_maps_correctly_with_example_labellers(use_gate, use_qubit, example_circuit, example_labellers):
    """Graph state mapper should map the circuit correctly with the passed labellers."""
    labellers = example_labellers
    circuit = example_circuit
    mapper = gseq.GraphStateEquivalenceMapper(use_gate_name=use_gate, use_qubits=use_qubit, op_labellers=labellers)
    gate = "_PauliX" if use_gate else "None"
    qb1 = "(cirq.LineQubit(0),)" if use_qubit else "None"
    qb2 = "(cirq.LineQubit(1),)" if use_qubit else "None"
    qb3 = "(cirq.LineQubit(3),)" if use_qubit else "None"
    should_be = (
        (gate, qb1, ("foo", 3)),
        (gate, qb2, ("foo", 3)),
        (gate, qb3, ("foo", 3)),
    )
    actual = mapper.graph_state_equiv_tuple(circuit)
    assert actual == should_be


def test_gs_mapper_maps_correctly_with_default_labellers(example_circuit):
    """Graph state mapper should map the circuit correctly with the default labellers."""
    circuit = example_circuit
    mapper = gseq.GraphStateEquivalenceMapper()
    should_be = (
        ("_PauliX", "(cirq.LineQubit(0),)", 0),
        ("_PauliX", "(cirq.LineQubit(1),)", 0),
        ("_PauliX", "(cirq.LineQubit(3),)", 0),
    )
    actual = mapper.graph_state_equiv_tuple(circuit)
    assert actual == should_be


@pytest.mark.parametrize(
    "angle,family", [(0.25, 0), (1.25, 1), (1.75, 2), (0.75, 3), (0.5, 4), (1.5, 5), (1, 6), (0, 7)]
)
@pytest.mark.parametrize("gate", [(cirq.Rz), (cirq.Ry), (cirq.Rx), (QubitizedRotation)])
def test_rotation_family_labels_correctly(gate, angle, family):
    """Rotation operations should be mapped to their correct families (by angle)."""
    if gate == QubitizedRotation:
        gate = gate(n_controls=1, rads=angle * np.pi)
        op = gate.on(*cirq.LineQubit.range(2))
    else:
        op = gate(rads=angle * np.pi).on(cirq.LineQubit(0))
    labeller = gseq.rotation_family
    actual = labeller(op)
    should_be = family
    assert actual == should_be


def test_pauli_lcu_block_encoding():
    """Pauli LCU block encoding labellers should label correctly."""
    model = getInstance("FermiHubbard", shape=(2, 2), J=-1.0, U=4.0, cell=SquareLattice)
    block_encoding_gate = getEncoding(VALID_ENCODINGS.PauliLCU)(model)
    qubits = cirq.LineQubit.range(block_encoding_gate.num_qubits())
    op = block_encoding_gate.on(*qubits)

    labeller = gseq.block_encoding
    actual = labeller(op)
    should_be = "Fermi-Hubbard Model - SquareLattice(regular)\n\r\n\tN:\t(2, 2)"
    assert actual == should_be


def test_pauli_lcu_prepare():
    """Pauli LCU block prepare gate labellers should label correctly."""
    selection_bitsize = 5
    coeffs = (1, 2, 3, -2)
    prep_gate = prepare_pauli_lcu(5, coeffs)
    op = prep_gate.on(*cirq.LineQubit.range(selection_bitsize))
    labeller = gseq.prepare_args
    actual = labeller(op)
    should_be = (selection_bitsize, coeffs)
    assert actual == should_be


def test_pauli_lcu_select():
    """Pauli LCU block select gate labellers should label correctly."""
    selection_bitsize = 5
    target_bitsize = 3
    control_val = 0
    select_unitaries = [cirq.DensePauliString("XXY"), cirq.DensePauliString("YYZ")]
    gate = qt.bloqs.multiplexers.select_pauli_lcu.SelectPauliLCU(
        selection_bitsize, target_bitsize, select_unitaries, control_val
    )
    op = gate.on(*cirq.LineQubit.range(gate.num_qubits()))
    labeller = gseq.select
    actual = labeller(op)
    should_be = (selection_bitsize, target_bitsize, ("+XXY", "+YYZ"), control_val)
    assert actual == should_be
