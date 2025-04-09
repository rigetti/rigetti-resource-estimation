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

"""Unit tests for the transpile module of rigetti-resource-estimation."""

from rigetti_resource_estimation.transpile import Transpiler


TEST_QASM = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        h q[2];
        // replacements start here
        cp(0) q[1],q[2];
        p(pi/4) q[1];
        p(1.25*pi) q[0];
        p(pi*-0.25) q[2];
        p(3*pi/4) q[1];
        p(--1*pi/2) q[0];
        p(pi*1.5) q[0];
        p(-1*pi/2) q[1];
        p(-1.5*pi) q[2];
        p(pi) q[0];
        p(--4*pi/2) q[0];
        cu1(pi/32) q[0],q[1];
        cp(pi) q[0],q[1];
        u1(-0.1*pi) q[0];
        u3(-pi/64,0.1*pi,3.02) q[2];
        ry(-pi/1.53) q[1];
        rx(pi*1.75) q[0];
        ccz q[0],q[1],q[2];
        ccx q[0],q[1],q[2];
        rz(1.25*pi) q[2];
        tdg q[1];
        // replacements finish here
        cz q[0],q[1];"""

transpiler = Transpiler(TEST_QASM)
transpiled_qasm_counts = transpiler.qasm_exact()


class TestTranspiler:
    """A class to test the Transpiler object and its methods."""

    def test_counting(self):
        """Pytests for simple_qasm_transpiler function."""
        assert transpiled_qasm_counts.init_rz_count == 8
        assert transpiled_qasm_counts.init_t_count == 21
        assert transpiled_qasm_counts.init_clifford_count == 10

    def test_exact_qasm_transpiler(self):
        """Pytests for exact_qasm_transpiler function."""
        expected_qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        h q[2];
        cx q[1],q[2];
        cx q[1],q[2];
        t q[1];
        z q[0];
        t q[0];
        tdg q[2];
        z q[1];
        tdg q[1];
        s q[0];
        z q[0];
        s q[0];
        z q[1];
        s q[1];
        s q[2];
        z q[0];
        rz(0.04908738521234052) q[0];
        cx q[0],q[1];
        rz(6.234097921967246) q[1];
        cx q[0],q[1];
        rz(0.04908738521234052) q[1];
        s q[0];
        cx q[0],q[1];
        z q[1];
        s q[1];
        cx q[0],q[1];
        s q[1];
        rz(5.969026041820607) q[0];
        rz(3.02) q[2];
        h q[2];
        s q[2];
        h q[2];
        rz(6.234097921967246) q[2];
        z q[2];
        h q[2];
        s q[2];
        h q[2];
        rz(0.31415926535897937) q[2];
        z q[2];
        z q[1];
        s q[1];
        h q[1];
        rz(4.229856775421551) q[1];
        h q[1];
        s q[1];
        h q[0];
        tdg q[0];
        h q[0];
        cx q[1],q[2];
        tdg q[2];
        cx q[0],q[2];
        t q[2];
        cx q[1],q[2];
        tdg q[2];
        cx q[0],q[2];
        t q[2];
        t q[1];
        cx q[0],q[1];
        t q[0];
        tdg q[1];
        cx q[0],q[1];
        h q[2];
        cx q[1],q[2];
        tdg q[2];
        cx q[0],q[2];
        t q[2];
        cx q[1],q[2];
        tdg q[2];
        cx q[0],q[2];
        t q[1];
        t q[2];
        h q[2];
        cx q[0],q[1];
        t q[0];
        tdg q[1];
        cx q[0],q[1];
        z q[2];
        t q[2];
        tdg q[1];
        cz q[0],q[1];"""

        expected_qasm += "\n"

        assert transpiled_qasm_counts.transpiled_qasm == expected_qasm
