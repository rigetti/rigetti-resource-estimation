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
**Module** ``rigetti_resource_estimation.translators``

A collection of operation translation tools to count, parse, and translate input logical gate sequences. These convert
operations in one framework into operations from another.
"""
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List, Any, Union

import numpy as np
import cirq
from qualtran.bloqs.basic_gates.rotation import ZPowGate as qt_zpowgate

from cabaliser import gates

CabaliserOp = tuple[int, tuple[int]]
Operation = Union[cirq.Operation, CabaliserOp]
Qubit = Union[cirq.Qid]  # type: ignore
Cirq2CabaliserFn = Callable[[cirq.Operation, Dict[cirq.Qid, int]], CabaliserOp]  # type: ignore

SELF_INVERSE_GATES = ["I", "X", "Y", "Z", "H", "CNOT", "CZ"]
DIRECT_CIRQ_TO_CABALISER_GATES = SELF_INVERSE_GATES + ["S"]


def is_clifford(angle, tol=1e-8) -> bool:
    """Helper function to determine if a z rotation, of passed angle, could be performed by Clifford gates.

    Note: passed angle is in rads, as a multiple of pi. That is, if the z rotation would be 3/5*pi rads, then the angle
    passed to this function would be 3/5.
    """
    return np.isclose(np.mod(angle, 2) / (1 / 4), round(np.mod(angle, 2) / (1 / 4)), atol=tol, rtol=0)


@dataclass
class OperationTranslator:
    """Translator class that converts operations from the 'from_framework' to the 'to_framework.'

    :param from_framework: The framework of the operation to convert (ex: cirq).
    :param to_framework: The framework of the operation to convert to (ex: cabaliser).
    :param op_id: A name or an identification for an translator instance.
    :param translate_fn: A function that performs the conversion from the original framework to the new framework.
    :param can_apply_to_fn: A function that checks if an operation can be converted by this translator instance.
    :param labels: list of tags to apply to this translator (ex: clifford). Useful for aggregation of stats.
    :param meta: User defined metadata for this translator.
    """

    from_framework: str
    to_framework: str
    op_id: Any
    translate_fn: Callable[[Operation, Dict[Qubit, int]], Operation]
    can_apply_to_fn: Callable[Operation, bool]  # type: ignore
    labels: List[str]
    meta: Dict[str, Any] = None  # type: ignore

    def __post_init__(self):
        self.reset_count()

    def reset_count(self):
        self.times_called = 0

    def convert(self, operation: Operation, qb_map: Dict[Qubit, int]) -> Operation:
        """Convert the operation in one circuit framework (ex: cirq) into another (ex: cabaliser).

        :param operation: The operation in the original framework.
        :param qb_map: A dictionary mapping qubits in the original framework to integers.
        """
        self.times_called += 1
        return self.translate_fn(operation, qb_map)

    def can_apply_to(self, op) -> bool:
        """Check to see if the passed operation can be converted by this translator.

        :param op: The operation to check.
        """
        return self.can_apply_to_fn(op)  # type: ignore


class CirqToCabaliser(OperationTranslator):
    """A translator that converts cirq operations to cabaliser operation tuples.

    For simplicity, default behavior is to use a cirq gate object as the op_id, and look-up the cabaliser gate.
    """

    def __init__(
        self,
        op_id: Any,
        translate_fn: Optional[Cirq2CabaliserFn] = None,
        can_apply_to_fn: Optional[Callable[cirq.Operation, bool]] = None,  # type: ignore
        labels: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if can_apply_to_fn is None:
            if not isinstance(op_id, cirq.Gate):  # type: ignore
                raise TypeError("For default behavior, op_id must be a cirq gate object.")
        translate_fn = translate_fn or self._simple_translate_fn
        can_apply_to_fn = can_apply_to_fn or self._default_apply_to_fn  # type: ignore
        super().__init__("cirq", "cabaliser", op_id, translate_fn, can_apply_to_fn, labels, meta)  # type: ignore

    def _get_cabaliser_gate(self, gate: cirq.Gate) -> int:  # type: ignore
        """Get the gate name that cabaliser uses.

        This is typically the same string (X, H, etc), but not for all gates; S**-1 gate, for example.
        """
        if gate == cirq.S**-1:
            return gates.Sd  # type: ignore
        return getattr(gates, str(gate))

    def _simple_translate_fn(self, op: cirq.Operation, qb_map: Dict[cirq.Qid, int]) -> tuple:  # type: ignore
        """Translates non-parameterized gates to cabaliser tuple based definition."""
        assert self.can_apply_to_fn(
            op  # type: ignore
        ), f"{op.gate} cannot be converted with this translator, which converts {self.op_id}"
        cabaliser_gate = self._get_cabaliser_gate(op.gate)  # type: ignore
        mapped_qbs = tuple(qb_map[qb] for qb in op.qubits)
        converted = (cabaliser_gate, mapped_qbs)
        return converted

    def _default_apply_to_fn(self, op: cirq.Operation) -> bool:  # type: ignore
        """If default behavior, the op_id is a cirq Gate. Checks the gate matches the op_id but is not an inverse."""
        if self.op_id == op.gate:
            if "**-1" not in str(op.gate):
                return True
            if op.gate == cirq.S**-1:
                return True
        return False


# Pre-constructed Translators
CIRQ2CABALISER_SIMPLE_TRANSLATORS = [
    CirqToCabaliser(getattr(cirq, gate), labels=["clifford"], meta={"angle_tag": None})
    for gate in DIRECT_CIRQ_TO_CABALISER_GATES
] + [
    CirqToCabaliser(cirq.S**-1, labels=["clifford"], meta={"angle_tag": None})  # type: ignore
]

# Self Inverse Translators
CIRQ2CABALISER_CNOT_INV = CirqToCabaliser(
    "CNOT**-1",
    lambda op, qbmap: (gates.CNOT, tuple(qbmap[qb] for qb in op.qubits)),  # type: ignore
    lambda op: str(op.gate) in ["CNOT**-1", "CNOT**-1.0"],  # type: ignore
    labels=["clifford"],
    meta={"angle_tag": None},
)
CIRQ2CABALISER_CZ_INV = CirqToCabaliser(
    "CZ**-1",
    lambda op, qbmap: (gates.CZ, tuple(qbmap[qb] for qb in op.qubits)),  # type: ignore
    lambda op: str(op.gate) in ["CZ**-1", "CZ**-1.0"],  # type: ignore
    labels=["clifford"],
    meta={"angle_tag": None},
)
CIRQ2CABALISER_H_INV = CirqToCabaliser(
    "H**-1",
    lambda op, qbmap: (gates.H, tuple(qbmap[qb] for qb in op.qubits)),  # type: ignore
    lambda op: str(op.gate) in ["H**-1", "H**-1.0"],  # type: ignore
    labels=["clifford"],
    meta={"angle_tag": None},
)
CIRQ2CABALISER_X_INV = CirqToCabaliser(
    "X**-1",
    lambda op, qbmap: (gates.X, tuple(qbmap[qb] for qb in op.qubits)),  # type: ignore
    lambda op: str(op.gate) in ["X**-1", "X**-1.0"],  # type: ignore
    labels=["clifford"],
    meta={"angle_tag": None},
)
CIRQ2CABALISER_Y_INV = CirqToCabaliser(
    "Y**-1",
    lambda op, qbmap: (gates.Y, tuple(qbmap[qb] for qb in op.qubits)),  # type: ignore
    lambda op: str(op.gate) in ["Y**-1", "Y**-1.0"],  # type: ignore
    labels=["clifford"],
    meta={"angle_tag": None},
)
CIRQ2CABALISER_Z_INV = CirqToCabaliser(
    "Z**-1",
    lambda op, qbmap: (gates.Z, tuple(qbmap[qb] for qb in op.qubits)),  # type: ignore
    lambda op: str(op.gate) in ["Z**-1", "Z**-1.0"],  # type: ignore
    labels=["clifford"],
    meta={"angle_tag": None},
)


CIRQ2CABALISER_SELF_INVERSE_TRANSLATORS = [
    CIRQ2CABALISER_CNOT_INV,
    CIRQ2CABALISER_CZ_INV,
    CIRQ2CABALISER_X_INV,
    CIRQ2CABALISER_Y_INV,
    CIRQ2CABALISER_Z_INV,
    CIRQ2CABALISER_H_INV,
]

# Other translators
# WARNING: X/Y/ZPowerGates also have a global phase. What should be done about it? For now, ignored.
CIRQ2CABALISER_ARB_Z = CirqToCabaliser(
    "ArbRz",
    lambda op, qbmap: (  # type: ignore
        gates.RZ,
        (qbmap[op.qubits[0]], 77),  # 77 arbitrary for now. This is tag for non-clifford angles.
    ),
    lambda op: (issubclass(type(op.gate), cirq.ZPowGate) or issubclass(type(op.gate), qt_zpowgate))
    and (not is_clifford(op.gate.exponent)),
    labels=["rz"],
    meta={"angle_tag": 77},
)
CIRQ2CABALISER_GLOBAL_PHASE = CirqToCabaliser(
    "GlobalPhase",
    lambda op, qbmap: (  # Below is a placeholder for all trivial ops for resource est purposes # type: ignore
        gates.I,
        (0,),
    ),
    lambda op: isinstance(op.gate, cirq.GlobalPhaseGate),  # type: ignore
    labels=["global phase"],
)
CIRQ2CABALISER_MEASUREMENT = CirqToCabaliser(
    "Measurement",
    lambda op, qbmap: (  # type: ignore
        gates.I,
        (0,),
    ),
    lambda op: isinstance(op.gate, cirq.MeasurementGate),  # type: ignore
    labels=["measurement"],
)
CIRQ2CABALISER_RESET = CirqToCabaliser(
    "Reset",
    lambda op, qbmap: (  # type: ignore
        gates.I,
        (0,),
    ),
    lambda op: isinstance(op.gate, cirq.ResetChannel),  # type: ignore
    labels=["reset"],
)
CIRQ2CABALISER_T = CirqToCabaliser(
    cirq.T,  # type: ignore
    lambda op, qbmap: (
        gates.RZ,
        (qbmap[op.qubits[0]], 31),
    ),  # 31 arbitrary for now. This is tag for T gate angles # type: ignore
    lambda op: op.gate == cirq.T,  # type: ignore
    labels=["t"],
    meta={"angle_tag": 31},
)
CIRQ2CABALISER_TDAG = CirqToCabaliser(
    cirq.T**-1,  # type: ignore
    lambda op, qbmap: (  # type: ignore
        gates.RZ,
        (qbmap[op.qubits[0]], 333),
    ),  # 333 arbitrary for now. This is the tag for T dagger angle.
    lambda op: op.gate == cirq.T**-1,  # type: ignore
    labels=["t"],
    meta={"angle_tag": 333},
)

CIRQ2CABALISER_OTHER_TRANSLATORS = [
    CIRQ2CABALISER_ARB_Z,
    CIRQ2CABALISER_GLOBAL_PHASE,
    CIRQ2CABALISER_MEASUREMENT,
    CIRQ2CABALISER_RESET,
    CIRQ2CABALISER_T,
    CIRQ2CABALISER_TDAG,
]

DEFAULT_TRANSLATORS = (
    CIRQ2CABALISER_SIMPLE_TRANSLATORS + CIRQ2CABALISER_OTHER_TRANSLATORS + CIRQ2CABALISER_SELF_INVERSE_TRANSLATORS
)
