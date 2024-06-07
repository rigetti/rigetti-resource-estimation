# Copyright 2022-2024 Rigetti & Co, LLC
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
import logging
from typing import NamedTuple

from sympy.parsing.sympy_parser import parse_expr
from numpy import pi

logger = logging.getLogger(__name__)

t_angles = 0.25 * pi
zt_angles = 1.25 * pi
tdg_zst_angles = 1.75 * pi
ztdg_st_angles = 0.75 * pi
s_zsdg_angles = 0.5 * pi
zs_sdg_angles = 1.5 * pi
z_angles = pi
TRIVIAL_ANGLES = 0

TranspiledQASMCounts = NamedTuple(
    "TranspiledQASMCounts",
    [
        ("transpiled_qasm", str),
        ("init_t_count", int),
        ("init_rz_count", int),
        ("init_clifford_count", int),
    ],
)


def rz_angle_selector(angle: str, q: str) -> TranspiledQASMCounts:
    """Select, count, and print the equivalent ops to rz(`angle`) q[`q`] gate.

    :param angle: the rotation angle
    :param q: qubit index

    :returns: A TranspiledQASMCounts object with attributes as follows.
        `transpiled_qasm`: a string representing the equivalent gate(s).
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
    return TranspiledQASMCounts(
        transpiled_qasm=output,
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

    def qasm_exact(self) -> TranspiledQASMCounts:
        """Transpile input QASM circuit replacing unsupported gates with valid and exactly equivalent ops.

        This converts the QASM to an form appropriate for downstream FT-compilers. Additionally, counts the number of
        T and arbitrary-angle Rz gates in the input circuit.

        :param circuit: an OPENQASM2.0 circuit.

        :returns: A TranspiledQASMCounts object with the attributes as follows.
            `transpiled_circuit`: The transpiled QASM circuit. This is the equivalent quantum circuit with some gates
                replaced using **exact** equivalences.
            `init_t_count`: Number of T and TDagger gates in the original circuit.
            `init_rot_count`: Number of small-angle rotation gates in the original circuit.
            `init_clifford_count`: Number of explicit Clifford gates in the original circuit.
        """
        output = self.transpiled_circuit
        tot_rot_count = 0
        tot_t_count = 0
        tot_clifford_count = 0

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
                tot_t_count += 1
            elif re.search(r"^ *sdg ", line) is not None:  # T and T^dagger gates
                output += "        z q[" + q + "];\n"
                output += "        s q[" + q + "];\n"
                tot_clifford_count += 1
            elif re.search(r"^ *(measure|barrier|post|reset|//)", line) is not None:
                continue  # measure,barrier,post ops and comments
            elif re.search(r"^ *(p|u1|rz)\(", line) is not None:  # Phase, u1, rz gates
                transpiled_qasm_counts = rz_angle_selector(angle, q)
                output += transpiled_qasm_counts.transpiled_qasm
                tot_t_count += transpiled_qasm_counts.init_t_count
                tot_rot_count += transpiled_qasm_counts.init_rz_count
                tot_clifford_count += transpiled_qasm_counts.init_clifford_count
            elif re.search(r"^ *(cu1|cp)\(", line) is not None:  # CP gate
                transpiled_qasm_counts1 = rz_angle_selector(angle + "/2", qq0)
                out1 = "        cx q[" + qq0 + "],q[" + qq1 + "];\n"
                transpiled_qasm_counts2 = rz_angle_selector("-" + angle + "/2", qq1)
                out2 = "        cx q[" + qq0 + "],q[" + qq1 + "];\n"
                transpiled_qasm_counts3 = rz_angle_selector(angle + "/2", qq1)
                output += (
                    transpiled_qasm_counts1.transpiled_qasm
                    + out1
                    + transpiled_qasm_counts2.transpiled_qasm
                    + out2
                    + transpiled_qasm_counts3.transpiled_qasm
                )
                tot_t_count += (
                    transpiled_qasm_counts1.init_t_count
                    + transpiled_qasm_counts2.init_t_count
                    + transpiled_qasm_counts3.init_t_count
                )
                tot_rot_count += (
                    transpiled_qasm_counts1.init_rz_count
                    + transpiled_qasm_counts2.init_rz_count
                    + transpiled_qasm_counts3.init_rz_count
                )
                tot_clifford_count += (
                    1
                    if transpiled_qasm_counts1.init_t_count
                    + transpiled_qasm_counts2.init_t_count
                    + transpiled_qasm_counts3.init_t_count
                    + transpiled_qasm_counts1.init_rz_count
                    + transpiled_qasm_counts2.init_rz_count
                    + transpiled_qasm_counts3.init_rz_count
                    == 0
                    else 0
                )
            elif re.search(r"^ *(u|u3|U)\(", line) is not None:  # u, u3, U gates
                transpiled_qasm_counts1 = rz_angle_selector(aaa2, q)
                out1 = "        h q[" + q + "];\n"
                out1 += "        s q[" + q + "];\n"
                out1 += "        h q[" + q + "];\n"
                transpiled_qasm_counts2 = rz_angle_selector(aaa0, q)
                out2 = "        z q[" + q + "];\n"
                out2 += "        h q[" + q + "];\n"
                out2 += "        s q[" + q + "];\n"
                out2 += "        h q[" + q + "];\n"
                transpiled_qasm_counts3 = rz_angle_selector(aaa1, q)
                out3 = "        z q[" + q + "];\n"
                output += (
                    transpiled_qasm_counts1.transpiled_qasm
                    + out1
                    + transpiled_qasm_counts2.transpiled_qasm
                    + out2
                    + transpiled_qasm_counts3.transpiled_qasm
                    + out3
                )
                tot_t_count += (
                    transpiled_qasm_counts1.init_t_count
                    + transpiled_qasm_counts2.init_t_count
                    + transpiled_qasm_counts3.init_t_count
                )
                tot_rot_count += (
                    transpiled_qasm_counts1.init_rz_count
                    + transpiled_qasm_counts2.init_rz_count
                    + transpiled_qasm_counts3.init_rz_count
                )
                tot_clifford_count += (
                    1
                    if transpiled_qasm_counts1.init_t_count
                    + transpiled_qasm_counts2.init_t_count
                    + transpiled_qasm_counts3.init_t_count
                    + transpiled_qasm_counts1.init_rz_count
                    + transpiled_qasm_counts2.init_rz_count
                    + transpiled_qasm_counts3.init_rz_count
                    == 0
                    else 0
                )
            elif re.search(r"^ *ry\(", line) is not None:  # ry(arbitrary-angle) gate
                out1 = "        z q[" + q + "];\n"
                out1 += "        s q[" + q + "];\n"
                out1 += "        h q[" + q + "];\n"
                transpiled_qasm_counts2 = rz_angle_selector(angle, q)
                out2 = "        h q[" + q + "];\n"
                out2 += "        s q[" + q + "];\n"
                output += out1 + transpiled_qasm_counts2.transpiled_qasm + out2
                tot_t_count += transpiled_qasm_counts2.init_t_count
                tot_rot_count += transpiled_qasm_counts2.init_rz_count
                tot_clifford_count += (
                    1 if transpiled_qasm_counts2.init_t_count + transpiled_qasm_counts2.init_rz_count == 0 else 0
                )
            elif re.search(r"^ *rx\(", line) is not None:  # rx(arbitrary-angle) gate
                out1 = "        h q[" + q + "];\n"
                transpiled_qasm_counts2 = rz_angle_selector(angle, q)
                out2 = "        h q[" + q + "];\n"
                output += out1 + transpiled_qasm_counts2.transpiled_qasm + out2
                tot_t_count += transpiled_qasm_counts2.init_t_count
                tot_rot_count += transpiled_qasm_counts2.init_rz_count
                tot_clifford_count += (
                    1 if transpiled_qasm_counts2.init_t_count + transpiled_qasm_counts2.init_rz_count == 0 else 0
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
                tot_t_count += 7
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
                tot_t_count += 7
            elif re.search(r"^ *sx ", line) is not None:  # sx gate
                output += "        h q[" + q + "];\n"
                output += "        s q[" + q + "];\n"
                output += "        h q[" + q + "];\n"
                tot_clifford_count += 1
            elif re.search(r"^ *sxdg ", line) is not None:  # sxdg gate
                output += "        h q[" + q + "];\n"
                output += "        z q[" + q + "];\n"
                output += "        s q[" + q + "];\n"
                output += "        h q[" + q + "];\n"
                tot_clifford_count += 1
            else:
                output += line + "\n"
                if re.search(r"^ *(OPENQASM|include|qreg)", line) is None:
                    tot_clifford_count += 1

        return TranspiledQASMCounts(
            transpiled_qasm=output,
            init_t_count=tot_t_count,
            init_rz_count=tot_rot_count,
            init_clifford_count=tot_clifford_count,
        )


def precompile_circuit(circuit_qasm: str) -> TranspiledQASMCounts:
    """Offer an easy-to-use function for the Transpile class to pre-compile the circuit and print additional info.

    :param circuit_qasm: the logical-level OpenQASM2.0 circuit to be transpiled.

    :returns:
        `circuit_qasm`: The transpiled qasm circuit. This is the equivalent quantum circuit with some gates
            replaced using **exact** equivalences.
        `orig_t_count`: Number of explicit T and TDagger gates in the original circuit.
        `arbitrary_rot_count`: Number of small-angle rotation gates in the original circuit.
    """
    transpiler = Transpiler(circuit_qasm)
    transpiled_qasm_counts = transpiler.qasm_exact()
    return transpiled_qasm_counts
