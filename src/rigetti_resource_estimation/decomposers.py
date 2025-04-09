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
**Module** ``rigetti_resource_estimation.decomposers``

A collection of circuit decomposition tools and helper methods.
"""
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List, Any, Literal, Iterable

from numpy import pi
import numpy as np
import cirq
from qualtran.cirq_interop._bloq_to_cirq import BloqAsCirqGate
from qualtran.bloqs.basic_gates.rotation import ZPowGate as qt_zpowgate
from rigetti_resource_estimation.translators import Operation, Qubit
from pyLIQTR.circuits.operators.AddMod import AddMod as pyLAM

DecomposerType = Literal["intercepting", "fallback"]


@dataclass
class OperationDecomposer:
    """Decomposer class that converts operations from into equivalent operations in the same framework.

    :param framework: The framework of the operation to convert (ex: cirq).
    :param op_id: A name or an identification for an translator instance.
    :param decompose_fn: A function that performs the conversion.
    :param can_apply_to_fn: A function that checks if an operation can be converted by this decomposer instance.
    :param labels: list of tags to apply to this decomposer (ex: clifford). Useful for aggregation of stats.
    :param meta: User defined metadata for this decomposer.
    """

    framework: str
    decomposer_type: DecomposerType
    decomposer_id: Any
    decompose_fn: Callable[[Operation, Dict[Qubit, int]], Operation]
    can_apply_to_fn: Callable[Operation, bool]  # type: ignore
    labels: List[str]
    meta: Dict[str, Any] = None  # type: ignore

    def __post_init__(self):
        self.times_called = 0

    def convert(self, operation: Operation) -> Operation:
        """Convert the operation in one circuit framework (ex: cirq) into an equivalent set of operations.

        :param operation: The original operation.
        """
        self.times_called += 1
        return self.decompose_fn(operation)  # type: ignore

    def can_apply_to(self, op) -> bool:
        """Check to see if the passed operation can be converted by this translator.

        :param op: The operation to check.
        """
        return self.can_apply_to_fn(op)  # type: ignore


def process_z_rot(op: cirq.Operation, tol=1e-8) -> cirq.Operation:  # type: ignore
    """Helper function to replace z rotations with clifford or T gates."""
    # replace z rotations with clifford or T gates if possible
    exponent = op.gate.exponent  # type: ignore
    qb = op.qubits[0]
    if np.isclose(np.mod(exponent, 2), 1, atol=tol, rtol=0):
        return cirq.Z(qb)  # type: ignore
    if np.isclose(np.mod(exponent, 2), 1 / 4, atol=tol, rtol=0):
        return cirq.T(qb)  # type: ignore
    if np.isclose(np.mod(exponent, 2), 7 / 4, atol=tol, rtol=0):
        return (cirq.T**-1)(qb)  # type: ignore
    if np.isclose(np.mod(exponent, 2), 1 / 2, atol=tol, rtol=0):
        return cirq.S(qb)  # type: ignore
    if np.isclose(np.mod(exponent, 2), 3 / 2, atol=tol, rtol=0):
        return (cirq.S**-1)(qb)  # type: ignore
    if np.isclose(np.mod(exponent, 2), 0, atol=tol, rtol=0) or np.isclose(np.mod(exponent, 2), 2, atol=tol, rtol=0):
        return cirq.I(qb)  # type: ignore
    return op


PROCESS_Z_ROT_DECOMP = OperationDecomposer(
    framework="cirq",
    decomposer_type="intercepting",
    decomposer_id="process_z_rot",
    decompose_fn=process_z_rot,  # type: ignore
    can_apply_to_fn=lambda op: isinstance(op.gate, cirq.ZPowGate) or isinstance(op.gate, qt_zpowgate),
    labels=["z_rot"],
    meta={},
)

MODKDECOMP = OperationDecomposer(
    framework="cirq",
    decomposer_type="intercepting",
    decomposer_id="modk",
    decompose_fn=lambda op: pyLAM(
        bitsize=op.gate.bitsize, add_val=op.gate.add_val, mod=op.gate.mod, cvs=op.gate.cvs
    ).on(*op.qubits),
    can_apply_to_fn=lambda op: str(op).startswith("AddConstantMod") or str(op).startswith("ModAddK"),
    labels=["modk"],
    meta={},
)

# Fallback decomp is called if standard decomp fails and op isn't kept.
# This covers the case of operations without decompositons (in cirq's target gate set) but are not in the target gate
# set of cabaliser


def ry_to_rz_decomp(op: cirq.Operation, tol=1e-8) -> cirq.Operation:  # type: ignore
    """Decompose an arbitrary y rotation to arbitrary z rotation."""
    exponent = op.gate.exponent  # type: ignore
    qb = op.qubits[0]
    new_ops = []
    new_ops.append(cirq.Z.on(qb))  # type: ignore
    new_ops.append(cirq.S.on(qb))  # type: ignore
    new_ops.append(cirq.H.on(qb))  # type: ignore
    z_rot = cirq.ZPowGate(exponent=exponent).on(qb)  # type: ignore
    new_ops.append(process_z_rot(z_rot, tol))
    new_ops.append(cirq.H.on(qb))  # type: ignore
    new_ops.append(cirq.S.on(qb))  # type: ignore
    return new_ops  # type: ignore


def rx_to_rz_decomp(op: cirq.Operation, tol=1e-8) -> List[cirq.Operation]:  # type: ignore
    """Decompose an arbitrary y rotation to arbitary z rotation."""
    exponent = op.gate.exponent  # type: ignore
    qb = op.qubits[0]
    new_ops = []
    new_ops.append(cirq.H.on(qb))  # type: ignore
    z_rot = cirq.ZPowGate(exponent=exponent).on(qb)  # type: ignore
    new_ops.append(process_z_rot(z_rot, tol))
    new_ops.append(cirq.H.on(qb))  # type: ignore
    return new_ops


def qualtran_bloq_as_cirq(op: cirq.Operation) -> cirq.Operation:  # type: ignore
    """Helper method to convert a qualtran CNOT bloq to cirq."""
    qubits = op.qubits
    if str(op.gate.bloq) == "CNOT":  # type: ignore
        return cirq.CNOT(*qubits)  # type: ignore
    if str(op.gate.bloq) == "T":  # type: ignore
        return cirq.T(*qubits)  # type: ignore
    if str(op.gate.bloq) == "XGate":  # type: ignore
        return cirq.X(*qubits)  # type: ignore
    if str(op.gate.bloq) == "YGate":  # type: ignore
        return cirq.Y(*qubits)  # type: ignore
    if str(op.gate.bloq) == "ZGate":  # type: ignore
        return cirq.Z(*qubits)  # type: ignore
    if str(op.gate.bloq) == "SGate":  # type: ignore
        return cirq.S(*qubits)  # type: ignore
    raise ValueError(
        f"bloq_as_cirq decomposer trying to decompose {op} but incapable. Add separate decomposer for this op."
    )


def classicallycontrolled_decomp(op: cirq.Operation) -> cirq.Operation:  # type: ignore
    """Helper method to change a classically controlled operation to a quantum controlled version."""
    return op.without_classical_controls()


def czpower_decomp(op: cirq.Operation) -> List[cirq.Operation]:  # type: ignore
    """Helper method to decompose a CZPowGate."""
    qb1, qb2 = op.qubits
    exponent = op.gate.exponent  # type: ignore
    global_shift = op.gate.global_shift  # type: ignore
    total_phase = exponent * pi * global_shift * 1j
    operations = []
    operations.append(cirq.GlobalPhaseGate(np.exp(total_phase)).on())  # type: ignore
    operations.append(cirq.ZPowGate(exponent=exponent / 2).on(qb1))  # type: ignore
    operations.append(cirq.CNOT(qb1, qb2))  # type: ignore
    operations.append(cirq.ZPowGate(exponent=-exponent / 2).on(qb2))  # type: ignore
    operations.append(cirq.CNOT(qb1, qb2))  # type: ignore
    operations.append(cirq.ZPowGate(exponent=exponent / 2).on(qb2))  # type: ignore
    return operations


CIRQ_RY_TO_RZ_DECOMP = OperationDecomposer(
    framework="cirq",
    decomposer_type="fallback",
    decomposer_id="ry_to_rz",
    decompose_fn=ry_to_rz_decomp,  # type: ignore
    can_apply_to_fn=lambda op: isinstance(op.gate, cirq.YPowGate),  # type: ignore
    labels=["ry_to_rz"],
    meta={},
)
CIRQ_RX_TO_RZ_DECOMP = OperationDecomposer(
    framework="cirq",
    decomposer_type="fallback",
    decomposer_id="rx_to_rz",
    decompose_fn=rx_to_rz_decomp,  # type: ignore
    can_apply_to_fn=lambda op: isinstance(op.gate, cirq.XPowGate),  # type: ignore
    labels=["rx_to_rz"],
    meta={},
)
CIRQ_CZPOWGATE_DECOMP = OperationDecomposer(
    framework="cirq",
    decomposer_type="fallback",
    decomposer_id="czpower",
    decompose_fn=czpower_decomp,  # type: ignore
    can_apply_to_fn=lambda op: isinstance(op.gate, cirq.CZPowGate),  # type: ignore
    labels=["rx_to_rz"],
    meta={},
)
CIRQ_CLASSIC_CONTROL_DECOMP = OperationDecomposer(
    framework="cirq",
    decomposer_type="fallback",
    decomposer_id="classical_control",
    decompose_fn=classicallycontrolled_decomp,  # type: ignore
    can_apply_to_fn=lambda op: isinstance(op, cirq.ClassicallyControlledOperation),  # type: ignore
    labels=["classical_control"],
    meta={},
)
CIRQ_QT_BLOQ_DECOMP = OperationDecomposer(
    framework="cirq",
    decomposer_type="fallback",
    decomposer_id="qualtran_block",
    decompose_fn=qualtran_bloq_as_cirq,  # type: ignore
    can_apply_to_fn=lambda op: isinstance(op.gate, BloqAsCirqGate),  # type: ignore
    labels=["qualtran_block"],
    meta={},
)

CIRQ_INTERCEPTING_DECOMPS = [PROCESS_Z_ROT_DECOMP, MODKDECOMP, CIRQ_CLASSIC_CONTROL_DECOMP]

CIRQ_FALLBACK_DECOMPS = [
    CIRQ_RX_TO_RZ_DECOMP,
    CIRQ_RY_TO_RZ_DECOMP,
    CIRQ_QT_BLOQ_DECOMP,
    CIRQ_CZPOWGATE_DECOMP,
]


class CirqDecomposer:
    def __init__(self, op_decomposers: Optional[List[OperationDecomposer]] = None) -> None:
        if op_decomposers:
            if not all(decomposer.framework == "cirq" for decomposer in op_decomposers):
                raise ValueError("""At least 1 decomposer framework is not 'cirq'.""")
            self.op_decomposers = {
                decomp_type: [decomp for decomp in op_decomposers if decomp.decomposer_type == decomp_type]
                for decomp_type in ["intercepting", "fallback"]
            }

    def decompose_and_qb_map(
        self, circuit: cirq.Circuit, keep=None, context: Optional[cirq.DecompositionContext] = None  # type: ignore
    ) -> tuple[Iterable[cirq.Operation], Dict[cirq.Qid, int]]:  # type: ignore
        """Decompose the circuit to operations in the alphabet, and return a mapping of all qubits in the circuit.

        :param circuit: Circuit to decompose and map.
        :params keep: callable function to determine if a gate should NOT be decomposed.
        :param context: decomposition context to handle the qubit management during decomposition.
        """
        decomposed = cirq.Circuit(  # type: ignore
            cirq.decompose(  # type: ignore
                circuit,
                keep=keep,
                fallback_decomposer=self._make_decomp_fn("fallback"),
                intercepting_decomposer=self._make_decomp_fn("intercepting"),
                context=context,
            )
        )
        qubit_map = {qb: i for i, qb in enumerate(decomposed.all_qubits())}
        return decomposed, qubit_map  # type: ignore

    def _make_decomp_fn(self, decomp_type):
        def fn(op):
            decomposed = [decomp.convert(op) for decomp in self.op_decomposers[decomp_type] if decomp.can_apply_to(op)]
            if len(decomposed) == 1:
                return decomposed[0]
            if len(decomposed) > 1:
                compatible_decomposers = [
                    decomposer for decomposer in self.op_decomposers[decomp_type] if decomposer.can_apply_to(op)
                ]
                raise ValueError(
                    f"{op} was converted by more than 1 decomposer. Compatible decomposers are {compatible_decomposers}"
                )

        return fn


DEFAULT_DECOMPOSERS = CIRQ_INTERCEPTING_DECOMPS + CIRQ_FALLBACK_DECOMPS
