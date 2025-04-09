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
**Module** ``examples/widgetization_example.py``

Examples for demonstrating widgetization.
"""
import rigetti_resource_estimation.gs_equivalence as gseq
from rigetti_resource_estimation import widgetization


def make_fh_circuit():
    """Helper function to build Fermi-Hubbard circuit for testing."""
    from pyLIQTR.ProblemInstances.getInstance import getInstance
    from pyLIQTR.clam.lattice_definitions import SquareLattice
    from pyLIQTR.BlockEncodings.getEncoding import getEncoding, VALID_ENCODINGS
    from pyLIQTR.qubitization.qsvt_dynamics import qsvt_dynamics, simulation_phases

    # Create Fermi-Hubbard Instance
    J = -1.0
    N = 2
    U = 8.0
    p_algo = 0.5
    eps = (1 - p_algo) / 2

    # encoding_type = "PauliLCU" #FermiHubbardSquare
    model = getInstance("FermiHubbard", shape=(N, N), J=J, U=U, cell=SquareLattice)
    times = 1.0
    scaled_times = times * model.alpha
    phases = simulation_phases(times=scaled_times, eps=eps, precompute=False, phase_algorithm="optimization")
    gate_qsvt = qsvt_dynamics(encoding=getEncoding(VALID_ENCODINGS.PauliLCU), instance=model, phase_sets=phases)
    return gate_qsvt.circuit


def make_random_circuit():
    """Helper function to build random circuit for testing."""
    import cirq.testing as ct  # pylint: disable=import-outside-toplevel

    return ct.random_circuit(qubits=10, n_moments=10, op_density=0.7)


def run_example(circuit_gen_fn=make_random_circuit):
    """Run a simple example of widgetizing a small circuit."""
    circuit = circuit_gen_fn()
    gs_equiv_mapper = gseq.GraphStateEquivalenceMapper(op_labellers=gseq.get_all_non_trivial_op_labeller_list())

    # Widgetize
    small_enough_fn = widgetization.small_enough_pyliqtr_re(max_gates=10000, max_q=100)
    widgetizer = widgetization.CirqWidgetizer(small_enough_fn=small_enough_fn, gs_equiv_mapper=gs_equiv_mapper)
    result = widgetizer.widgetize(circuit)

    ## To write as QASM use:
    # writer = QASMWidgetWriter()
    # as_qasm = writer.write(result)
    return result
