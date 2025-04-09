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
**Module** ``rigetti_resource_estimation.gs_equivalence``

Tools for labelling quantum operations and evaluating if circuits are graph state equivalent.
"""
from typing import Callable, Union, Optional, Dict
import cirq
import pyLIQTR
from pyLIQTR.qubitization.qubitized_gates import QubitizedRotation
from pyLIQTR.BlockEncodings.PauliStringLCU import PauliStringLCU
import numpy as np
import qualtran as qt

#: Labels for families of rotation angles in units of pi radians. For example, if a rotation has an angle of 0.25pi
# radians, it belongs to rotation family 0.
ROTATION_FAMILIES = {0.25: 0, 1.25: 1, 1.75: 2, 0.75: 3, 0.5: 4, 1.5: 5, 1: 6, 0: 7}
CircuitGSEquivTuple = tuple[tuple[Union[int, float, str, bool]]]

# note: CircuitOperation operations return None for their op.Gate.
OpLabellerDict = Dict[Optional[cirq.Gate], Callable[[cirq.Operation], CircuitGSEquivTuple]]


def qubits(op: cirq.Operation) -> str:
    """Returns qubits that the operations is acting on."""
    return str(op.qubits)


def gate_class(op: cirq.Operation) -> str:
    """Returns class the operation's gate.

    :param op: operation to be labelled with this function.
    """
    return type(op.gate).__name__


def rotation_family(op: cirq.ops.Operation, tolerance: float = 1e-6) -> Optional[int]:
    """If op is a rotation, return its angle family. If not, return None.

    :param op: operation to be labelled with this function.
    """
    rotation_gates = [QubitizedRotation, cirq.Rx, cirq.Ry, cirq.Rz]
    if type(op.gate) not in rotation_gates:
        return None
    rad_family = 8
    for rot in ROTATION_FAMILIES.keys():
        if np.isclose((op.gate._rads / np.pi) % 2, rot):
            rad_family = ROTATION_FAMILIES[rot]
    return rad_family


def block_encoding(op: cirq.ops.Operation) -> Optional[str]:
    """If op is a PauliLCU block encoding, return its properties (stored in PI attribute).

    :param op: operation to be labelled with this function.
    """
    return str(op.gate.PI)


def prepare_args(op: cirq.ops.Operation) -> Optional[tuple[int, tuple[float]]]:
    """If op is a PauliLCU Prepare gate, return its properties.

    :param op: operation to be labelled with this function.
    """
    return (op.gate._selection_bitsize, tuple(op.gate._alphas))


def select(op: cirq.ops.Operation) -> Optional[tuple[int, int, tuple[str], int]]:
    """If op is a PauliLCU Prepare gate, return its properties.

    :param op: operation to be labelled with this function.
    """
    unitary_strings = tuple(str(unitary) for unitary in op.gate.select_unitaries)
    return (op.gate.selection_bitsize, op.gate.target_bitsize, unitary_strings, op.gate.control_val)


SUPPORTED_GATES = {
    QubitizedRotation: rotation_family,
    cirq.Rx: rotation_family,
    cirq.Ry: rotation_family,
    cirq.Rz: rotation_family,
    PauliStringLCU: block_encoding,
    pyLIQTR.circuits.operators.select_prepare_pauli.prepare_pauli_lcu: prepare_args,
    qt.bloqs.multiplexers.select_pauli_lcu.SelectPauliLCU: select,
}


class GraphStateEquivalenceMapper:
    """A class that orchestrates turning a cirq circuit into a tuple used for comparison of expected graph states.

    If two circuits have are converted to the same tuple, they are expected to have the same graph state, and related
    properties after compilation. Each instance of GraphStateEquivalenceMapper is equipped with a list of op_labellers,
    which are callable functions that add additional labels to the tuple.
    """

    def __init__(
        self,
        use_gate_name: bool = True,
        use_qubits: bool = True,
        op_labellers: Optional[OpLabellerDict] = None,
    ):
        """
        :param use_gate_name: flag to use gate names to indicate different graph states (diff gates are not gs equiv).
        :param use_qubits: flag to use qubits to indicate different graph states (diff qubits are not gs equiv).
        :param op_labellers: list of callable functions that map the circuit to tuple entries.
        """
        self.gate_name = gate_class if use_gate_name else lambda _: "None"
        self.qubits = qubits if use_qubits else lambda _: "None"
        self.op_labellers = {}
        if op_labellers is None:
            op_labellers = SUPPORTED_GATES
        self.op_labellers.update(op_labellers)

    def graph_state_equiv_tuple(self, circuit: cirq.Circuit) -> CircuitGSEquivTuple:
        """Perform the mapping of a cirq circuit to tuple according to the op_labellers.

        :param circuit: circuit to convert to a tuple.
        """
        ops = sorted(circuit.all_operations(), key=lambda op: op.qubits)
        return tuple(
            (
                self.gate_name(op),
                self.qubits(op),
                self.op_labellers[op.gate](op) if op.gate in self.op_labellers else 0,
            )
            for op in ops
        )
