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
**Module** ``rigetti_resource_estimation.transpile``

A collection of circuit transpilation tools to count, parse, and transpile input logical gate sequences. These replace
unsupported gates with equivalents for downstream compilers while complying with any hardware requirements.
"""

import re
from dataclasses import dataclass
import logging
from typing import Optional, Dict, List, Any

from sympy.parsing.sympy_parser import parse_expr
from numpy import pi
import cirq
import rigetti_resource_estimation.translators as translators

logger = logging.getLogger(__name__)

t_angles = 0.25 * pi
zt_angles = 1.25 * pi
tdg_zst_angles = 1.75 * pi
ztdg_st_angles = 0.75 * pi
s_zsdg_angles = 0.5 * pi
zs_sdg_angles = 1.5 * pi
z_angles = pi
TRIVIAL_ANGLES = 0


@dataclass
class TranspiledCirqCounts:
    """Results class to hold the data coming from the transpiler."""

    transpiled_cirq: str
    init_t_count: int
    init_rz_count: int
    init_clifford_count: int
    input_qubits: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


def rz_angle_selector(angle: str, q: str) -> TranspiledCirqCounts:
    """Select, count, and print the equivalent ops to rz(`angle`) q[`q`] gate.

    :param angle: the rotation angle
    :param q: qubit index

    :returns: A TranspiledCirqCounts object with attributes as follows.
        `transpiled_cirq`: a string representing the equivalent gate(s).
        `init_t_count`: total number of explicit T and TDagger gates.
        `init_rot_count`: total number of nontrivial (small-angle) Rz gates.
        `init_clifford_count`: total number of explicit Clifford gates.
    """
    output = ""
    rot_count = 0
    t_count = 0
    clifford_count = 0
    angle_f = float(parse_expr(angle)) % (2 * pi)
    if angle_f == t_angles:
        output += "        t q[" + q + "];\n"
        t_count += 1
    elif angle_f == zt_angles:
        output += "        z q[" + q + "];\n"
        output += "        t q[" + q + "];\n"
        t_count += 1
    elif angle_f == tdg_zst_angles:
        output += "        tdg q[" + q + "];\n"
        t_count += 1
    elif angle_f == ztdg_st_angles:
        output += "        z q[" + q + "];\n"
        output += "        tdg q[" + q + "];\n"
        t_count += 1
    elif angle_f == s_zsdg_angles:
        output += "        s q[" + q + "];\n"
        clifford_count += 1
    elif angle_f == zs_sdg_angles:
        output += "        z q[" + q + "];\n"
        output += "        s q[" + q + "];\n"
        clifford_count += 1
    elif angle_f == z_angles:
        output += "        z q[" + q + "];\n"
        clifford_count += 1
    elif angle_f != TRIVIAL_ANGLES:
        output += "        rz(" + str(angle_f) + ") q[" + q + "];\n"
        rot_count += 1
    return TranspiledCirqCounts(
        transpiled_cirq=output,
        init_t_count=t_count,
        init_rz_count=rot_count,
        init_clifford_count=clifford_count,
    )


class Transpiler:
    """A class that hosts transpiling and parsing methods to count and replace unsupported gates

    These methods ensure the input logical circuit is appropriate for downstream compilers.
    """

    def __init__(self, circuit: str):
        """:param circuit: the logical OpenQASM2.0 circuit."""
        self.input_circuit = circuit
        self.transpiled_circuit = ""  # initializing a shared transpiled_circuit variable for methods below.

    def qasm_exact(self) -> TranspiledCirqCounts:
        """Transpile input QASM circuit replacing unsupported gates with valid and exactly equivalent ops.

        This converts the QASM to an form appropriate for downstream FT-compilers. Additionally, counts the number of
        T and arbitrary-angle Rz gates in the input circuit.

        :param circuit: an OPENQASM2.0 circuit.

        :returns: A TranspiledCirqCounts object with the attributes as follows.
            `transpiled_circuit`: The transpiled QASM circuit. This is the equivalent quantum circuit with some gates
            replaced using **exact** equivalences.
            `init_t_count`: Number of T and TDagger gates in the original circuit.
            `init_rot_count`: Number of small-angle rotation gates in the original circuit.
            `init_clifford_count`: Number of explicit Clifford gates in the original circuit.
        """
        output = self.transpiled_circuit
        total_rot_count = 0
        total_t_count = 0
        total_clifford_count = 0

        for line in self.input_circuit.splitlines():
            match = re.search(r"\((.+)\)", line)
            angle = match.group(1) if match else ""

            match = re.search(r"\((.+),.+,", line)
            aaa0 = match.group(1) if match else ""
            match = re.search(r",(.+),", line)
            aaa1 = match.group(1) if match else ""
            match = re.search(r",.+,(.+)\)", line)
            aaa2 = match.group(1) if match else ""

            match = re.search(r"q\[(.+)\];", line)
            q = match.group(1) if match else ""

            match = re.search(r"q\[(.+)\],", line)
            qq0 = match.group(1) if match else ""
            match = re.search(r",q\[(.+)\];", line)
            qq1 = match.group(1) if match else ""

            match = re.search(r" q\[(.+)\],q\[.+\],q", line)
            qqq0 = match.group(1) if match else ""
            match = re.search(r",q\[(.+)\],q", line)
            qqq1 = match.group(1) if match else ""
            match = re.search(r",q\[.+\],q\[(.+)\];", line)
            qqq2 = match.group(1) if match else ""

            if re.search(r"^ *(t|tdg) ", line) is not None:  # T and T^dagger gates
                output += line + "\n"
                total_t_count += 1
            elif re.search(r"^ *sdg ", line) is not None:  # T and T^dagger gates
                output += "        z q[" + q + "];\n"
                output += "        s q[" + q + "];\n"
                total_clifford_count += 1
            elif re.search(r"^ *(measure|barrier|post|reset|//)", line) is not None:
                continue  # measure,barrier,post ops and comments
            elif re.search(r"^ *(p|u1|rz)\(", line) is not None:  # Phase, u1, rz gates
                transpiled_cirq_counts = rz_angle_selector(angle, q)
                output += transpiled_cirq_counts.transpiled_cirq
                total_t_count += transpiled_cirq_counts.init_t_count
                total_rot_count += transpiled_cirq_counts.init_rz_count
                total_clifford_count += transpiled_cirq_counts.init_clifford_count
            elif re.search(r"^ *(cu1|cp)\(", line) is not None:  # CP gate
                transpiled_cirq_counts1 = rz_angle_selector(angle + "/2", qq0)
                out1 = "        cx q[" + qq0 + "],q[" + qq1 + "];\n"
                transpiled_cirq_counts2 = rz_angle_selector("-" + angle + "/2", qq1)
                out2 = "        cx q[" + qq0 + "],q[" + qq1 + "];\n"
                transpiled_cirq_counts3 = rz_angle_selector(angle + "/2", qq1)
                output += (
                    transpiled_cirq_counts1.transpiled_cirq
                    + out1
                    + transpiled_cirq_counts2.transpiled_cirq
                    + out2
                    + transpiled_cirq_counts3.transpiled_cirq
                )
                total_t_count += (
                    transpiled_cirq_counts1.init_t_count
                    + transpiled_cirq_counts2.init_t_count
                    + transpiled_cirq_counts3.init_t_count
                )
                total_rot_count += (
                    transpiled_cirq_counts1.init_rz_count
                    + transpiled_cirq_counts2.init_rz_count
                    + transpiled_cirq_counts3.init_rz_count
                )
                total_clifford_count += (
                    1
                    if transpiled_cirq_counts1.init_t_count
                    + transpiled_cirq_counts2.init_t_count
                    + transpiled_cirq_counts3.init_t_count
                    + transpiled_cirq_counts1.init_rz_count
                    + transpiled_cirq_counts2.init_rz_count
                    + transpiled_cirq_counts3.init_rz_count
                    == 0
                    else 0
                )
            elif re.search(r"^ *(u|u3|U)\(", line) is not None:  # u, u3, U gates
                transpiled_cirq_counts1 = rz_angle_selector(aaa2, q)
                out1 = "        h q[" + q + "];\n"
                out1 += "        s q[" + q + "];\n"
                out1 += "        h q[" + q + "];\n"
                transpiled_cirq_counts2 = rz_angle_selector(aaa0, q)
                out2 = "        z q[" + q + "];\n"
                out2 += "        h q[" + q + "];\n"
                out2 += "        s q[" + q + "];\n"
                out2 += "        h q[" + q + "];\n"
                transpiled_cirq_counts3 = rz_angle_selector(aaa1, q)
                out3 = "        z q[" + q + "];\n"
                output += (
                    transpiled_cirq_counts1.transpiled_cirq
                    + out1
                    + transpiled_cirq_counts2.transpiled_cirq
                    + out2
                    + transpiled_cirq_counts3.transpiled_cirq
                    + out3
                )
                total_t_count += (
                    transpiled_cirq_counts1.init_t_count
                    + transpiled_cirq_counts2.init_t_count
                    + transpiled_cirq_counts3.init_t_count
                )
                total_rot_count += (
                    transpiled_cirq_counts1.init_rz_count
                    + transpiled_cirq_counts2.init_rz_count
                    + transpiled_cirq_counts3.init_rz_count
                )
                total_clifford_count += (
                    1
                    if transpiled_cirq_counts1.init_t_count
                    + transpiled_cirq_counts2.init_t_count
                    + transpiled_cirq_counts3.init_t_count
                    + transpiled_cirq_counts1.init_rz_count
                    + transpiled_cirq_counts2.init_rz_count
                    + transpiled_cirq_counts3.init_rz_count
                    == 0
                    else 0
                )
            elif re.search(r"^ *ry\(", line) is not None:  # ry(arbitrary-angle) gate
                out1 = "        z q[" + q + "];\n"
                out1 += "        s q[" + q + "];\n"
                out1 += "        h q[" + q + "];\n"
                transpiled_cirq_counts2 = rz_angle_selector(angle, q)
                out2 = "        h q[" + q + "];\n"
                out2 += "        s q[" + q + "];\n"
                output += out1 + transpiled_cirq_counts2.transpiled_cirq + out2
                total_t_count += transpiled_cirq_counts2.init_t_count
                total_rot_count += transpiled_cirq_counts2.init_rz_count
                total_clifford_count += (
                    1 if transpiled_cirq_counts2.init_t_count + transpiled_cirq_counts2.init_rz_count == 0 else 0
                )
            elif re.search(r"^ *rx\(", line) is not None:  # rx(arbitrary-angle) gate
                out1 = "        h q[" + q + "];\n"
                transpiled_cirq_counts2 = rz_angle_selector(angle, q)
                out2 = "        h q[" + q + "];\n"
                output += out1 + transpiled_cirq_counts2.transpiled_cirq + out2
                total_t_count += transpiled_cirq_counts2.init_t_count
                total_rot_count += transpiled_cirq_counts2.init_rz_count
                total_clifford_count += (
                    1 if transpiled_cirq_counts2.init_t_count + transpiled_cirq_counts2.init_rz_count == 0 else 0
                )
            elif re.search(r"^ *ccz(_\d+)? ", line) is not None:  # ccz and ccz(_digits) gates
                output += "        cx q[" + qqq1 + "],q[" + qqq2 + "];\n"
                output += "        tdg q[" + qqq2 + "];\n"
                output += "        cx q[" + qqq0 + "],q[" + qqq2 + "];\n"
                output += "        t q[" + qqq2 + "];\n"
                output += "        cx q[" + qqq1 + "],q[" + qqq2 + "];\n"
                output += "        tdg q[" + qqq2 + "];\n"
                output += "        cx q[" + qqq0 + "],q[" + qqq2 + "];\n"
                output += "        t q[" + qqq2 + "];\n"
                output += "        t q[" + qqq1 + "];\n"
                output += "        cx q[" + qqq0 + "],q[" + qqq1 + "];\n"
                output += "        t q[" + qqq0 + "];\n"
                output += "        tdg q[" + qqq1 + "];\n"
                output += "        cx q[" + qqq0 + "],q[" + qqq1 + "];\n"
                total_t_count += 7
            elif re.search(r"^ *ccx ", line) is not None:  # ccx gates
                output += "        h q[" + qqq2 + "];\n"
                output += "        cx q[" + qqq1 + "],q[" + qqq2 + "];\n"
                output += "        tdg q[" + qqq2 + "];\n"
                output += "        cx q[" + qqq0 + "],q[" + qqq2 + "];\n"
                output += "        t q[" + qqq2 + "];\n"
                output += "        cx q[" + qqq1 + "],q[" + qqq2 + "];\n"
                output += "        tdg q[" + qqq2 + "];\n"
                output += "        cx q[" + qqq0 + "],q[" + qqq2 + "];\n"
                output += "        t q[" + qqq1 + "];\n"
                output += "        t q[" + qqq2 + "];\n"
                output += "        h q[" + qqq2 + "];\n"
                output += "        cx q[" + qqq0 + "],q[" + qqq1 + "];\n"
                output += "        t q[" + qqq0 + "];\n"
                output += "        tdg q[" + qqq1 + "];\n"
                output += "        cx q[" + qqq0 + "],q[" + qqq1 + "];\n"
                total_t_count += 7
            elif re.search(r"^ *sx ", line) is not None:  # sx gate
                output += "        h q[" + q + "];\n"
                output += "        s q[" + q + "];\n"
                output += "        h q[" + q + "];\n"
                total_clifford_count += 1
            elif re.search(r"^ *sxdg ", line) is not None:  # sxdg gate
                output += "        h q[" + q + "];\n"
                output += "        z q[" + q + "];\n"
                output += "        s q[" + q + "];\n"
                output += "        h q[" + q + "];\n"
                total_clifford_count += 1
            elif re.search(r"^ *ch ", line) is not None:  # ch gate
                output += "        s q[" + qq1 + "];\n"
                output += "        h q[" + qq1 + "];\n"
                output += "        t q[" + qq1 + "];\n"
                output += "        cx q[" + qq0 + "],q[" + qq1 + "];\n"
                output += "        tdg q[" + qq1 + "];\n"
                output += "        h q[" + qq1 + "];\n"
                output += "        z q[" + qq1 + "];\n"
                output += "        s q[" + qq1 + "];\n"
                total_t_count += 2
            else:
                output += line + "\n"
                if re.search(r"^ *(OPENQASM|include|qreg)", line) is None:
                    total_clifford_count += 1

        return TranspiledCirqCounts(
            transpiled_cirq=output,
            init_t_count=total_t_count,
            init_rz_count=total_rot_count,
            init_clifford_count=total_clifford_count,
        )


def precompile_circuit(circuit_qasm: str) -> TranspiledCirqCounts:
    """Offer an easy-to-use function for the Transpile class to pre-compile the circuit and print additional info.

    :param circuit_qasm: the logical-level OpenQASM2.0 circuit to be transpiled.
    """
    transpiler = Transpiler(circuit_qasm)
    transpiled_cirq_counts = transpiler.qasm_exact()
    return transpiled_cirq_counts


class CirqTranspiler:
    """Transpiler acting on circuits generated with Cirq."""

    def __init__(self, translator_list: Optional[List[translators.CirqToCabaliser]] = None) -> None:
        """Translators are optional for backwards compatibility with existing Jablizer transpiler."""
        self.translators = translator_list or []

        # Ensure all translators are the same type (from and to) and from is always cirq
        if translator_list:
            if (len(set(translator.to_framework for translator in translator_list)) == 1) and all(
                translator.from_framework == "cirq" for translator in translator_list
            ):
                self._alphabet = [translator.op_id for translator in translator_list]
            else:
                raise ValueError(
                    """Translators inconsistent. At least 1 has to_framework different than the others, or
                    from_framework not cirq."""
                )

    @property
    def labels(self):
        """Return the set of all labels attached to translator."""
        return set(
            label
            for translator in self.translators
            for label in translator.labels  # pylint: disable=E1133 # type: ignore
        )

    @property
    def alphabet(self) -> List[Any]:
        """Return a list of op_ids of the translators."""
        return self._alphabet

    def reset_counts(self):
        [translator.reset_count() for translator in self.translators]

    def times_called(self, label: str) -> int:
        """Return the number of times an translator was called with the passed label.

        :param tag: Tag to query.
        """
        return sum((translator.times_called for translator in self.translators if label in translator.labels))

    def transpile(self, circuit: cirq.Circuit, qb_map: Optional[Dict[cirq.Qid, int]] = None) -> TranspiledCirqCounts:
        """Transpile the passed circuit into a TranspiledCirqCounts object for downstream RRE processing.

        Note: for backwards compatibility, this is still called a "TranspiledCirqCounts" object. This is likely to be
        updated away from referencing QASMs specifcially.

        :param circuit: The cirq circuit to transpile.
        :param qb_map: A dictionary containing the mapping of cirq qubits to integers.
        """
        self.reset_counts()
        if self.translators:
            return self._transpile_circuit(circuit, qb_map)
        return precompile_circuit(circuit)  # type: ignore

    def convert_op(self, op: cirq.Operation, qb_map: Dict[cirq.Qid, int]) -> translators.CabaliserOp:  # type: ignore
        """Convert the cirq operation to its cabaliser equivalent.

        :param op: The operation to convert.
        :param qb_map: A dictionary containing the mapping of cirq qubits to integers.
        """
        converted = [
            translator.convert(op, qb_map)
            for translator in self.translators
            if translator.can_apply_to(op)  # pylint: disable=E1133 # type: ignore
        ]
        if len(converted) == 1:
            return converted[0]  # type: ignore
        if len(converted) > 1:
            compatible_translators = [
                translator
                for translator in self.translators
                if translator.can_apply_to(op)  # pylint: disable=E1133 # type: ignore
            ]
            raise ValueError(
                f"{op} was converted by more than one translator. Compatible translators are {compatible_translators}"
            )
        raise ValueError(f"{op} could not be converted using existing translator.")

    def is_compatible(self, op: cirq.Operation) -> bool:  # type: ignore
        """Determines if this operator is compatible with any of the transpilers translators.

        :param op: Operation to check.
        """
        translator_can_handle = any(
            translator.can_apply_to(op) for translator in self.translators  # pylint: disable=E1133 # type: ignore
        )
        return translator_can_handle or isinstance(op.gate, cirq.GlobalPhaseGate)  # type: ignore

    def _transpile_circuit(self, circuit, qb_map=None):
        if not qb_map:
            qb_map = {qb: i for i, qb in enumerate(circuit.all_qubits())}
        operations = circuit.all_operations()
        converted = [self.convert_op(op, qb_map) for op in operations]
        counts = {label: self.times_called(label) for label in self.labels}
        input_qubits = len(qb_map)
        metadata = {
            str(translator.op_id): translator.meta
            for translator in self.translators  # pylint: disable=E1133 # type: ignore
        }

        return TranspiledCirqCounts(
            transpiled_cirq=converted,  # type: ignore
            init_t_count=counts["t"],
            init_rz_count=counts["rz"],
            init_clifford_count=counts["clifford"],
            input_qubits=input_qubits,
            metadata=metadata,
        )


def transpile_example():
    """Run a simple example of transpiling a small random circuit."""
    import cirq.testing as ct  # pylint: disable=import-outside-toplevel
    from rigetti_resource_estimation import decomposers

    op_translators = translators.CIRQ2CABALISER_SIMPLE_TRANSLATORS + translators.CIRQ2CABALISER_OTHER_TRANSLATORS
    transpiler = CirqTranspiler(translator_list=op_translators)
    keep_fn = transpiler.is_compatible

    rc = ct.random_circuit(qubits=5, n_moments=3, op_density=0.7)  # type: ignore
    print(rc)
    op_decomposers = decomposers.CIRQ_INTERCEPTING_DECOMPS + decomposers.CIRQ_FALLBACK_DECOMPS
    circuit_decomposer = decomposers.CirqDecomposer(op_decomposers)
    decomposed, _ = circuit_decomposer.decompose_and_qb_map(rc, keep=keep_fn)
    transpiled = transpiler.transpile(decomposed)  # type: ignore
    print(transpiled)
