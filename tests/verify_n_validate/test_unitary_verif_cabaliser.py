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
A collection of unit tests to robustly verify the unitariness of graph states and schedules outputted
by RRE and Cabaliser. The aim is to verify physical correctness of full set of dict/JSON Cabaliser's 
outputs considering widgetization and stitching. See references below for more details.

[1] S. N. Saadatmand, et al., Fault-tolerant resource estimation using graph-state compilation on a
modular superconducting architecture, arXiv:2406.06015 (2024, RRE original manuscript).
[2] https://github.com/Alan-Robertson/cabaliser
"""

import cirq
import json
from cabaliser import local_simulator

from rigetti_resource_estimation.estimation_pipeline import estimation_pipeline


def make_cabalize_dict_compatible(cabalize_dict: dict) -> dict:
    """Replace strings with equivalent integers for some fields of cabalize_dict making it backward compatible."""
    obj = cabalize_dict
    obj["adjacencies"] = {int(key): value for key, value in cabalize_dict["adjacencies"].items()}
    consump_shed = []
    for layer in cabalize_dict["consumptionschedule"]:
        subshed = []
        for layer_dict in layer:
            corrected_dict = {int(key): value for key, value in layer_dict.items()}
            subshed.append(corrected_dict)
        consump_shed.append(subshed)
    obj["consumptionschedule"] = consump_shed
    return obj


def test_unitariness_verif_cabaliser_qft3toffoli3():
    """
    A unitariness verification test for estimation_pipeline() and Cabaliser compiler running QFT3 and Toffoli algos.

    We start from a three-qubit all-zero state and apply a widgetized algorithm using the frontend
    estimation_pipeline(), default Cabaliser FT-compiler, and an RRE-formatted widgetized circuit dict. The input dict
    contains two example widgets with various number and types of operations. First widget contains a collection of
    clifford and non-clifford operations (note that too many non-clifford operations will make the exact simulations of
    graph state impossible), while the second one is a simple T+CNOT operation. We then, fully independent from
    estimation_pipeline() operations, apply the inverse of the widgets on the input state. If estimation_pipeline() and
    Cabaliser operations are valid by unitary principles we must measure an all-zeros state again. This simple test
    verifies the underlying widgetizition, transpilation, compilation, stitching, and post-processing operations of RRE.
    """
    compiler_tag_table = {0: local_simulator.I, 31: local_simulator.T, 77: local_simulator.T, 333: local_simulator.Tdag}

    circ_name = "verif_double_widgets"
    output_dir = "./output/" + circ_name + "/"
    cabalize_json0 = output_dir + circ_name + "_W0_all0init_cabalizeframes.json"
    cabalize_json1 = output_dir + circ_name + "_W1_all0init_cabalizeframes.json"

    n_qubits = 3
    qq = cirq.LineQubit.range(n_qubits)
    nonclifford_subcirc0 = cirq.inverse(  # Using cirq.inverse make it easy to apply the independent inverse ops later
        cirq.Circuit(
            cirq.Z(qq[0]),
            cirq.H(qq[1]),
            cirq.S(qq[2]),
            cirq.SWAP(qq[2], qq[1]),
            cirq.rz(-0.303)(qq[0]),
            cirq.Y(qq[1]),
            cirq.X(qq[2]),
            cirq.CX(qq[0], qq[1]),
            cirq.T(qq[1]),
            cirq.S(qq[2]) ** -1,
            cirq.Z(qq[0]),
            cirq.CZ(qq[0], qq[2]),
        )
    )
    nonclifford_subcirc1 = cirq.inverse(
        cirq.Circuit(
            cirq.T(qq[2]) ** -1,
            cirq.CX(qq[0], qq[1]),
            cirq.I(qq[2]),
        )
    )
    widgetized_circ_dict = {
        "stitches": {("W0", "W1"): 1},
        "widgets": {"W0": (nonclifford_subcirc0, 1), "W1": (nonclifford_subcirc1, 1)},
        "first_widget": "W0",
        "compiler_tag_table": {},
        "circuit_name": circ_name,
        "input_qubits": n_qubits,
        "init_t_count": None,
        "init_rz_count": None,
        "init_clifford_count": None,
    }

    estimation_pipeline(
        widgetized_circ_dict=widgetized_circ_dict,
        log="DEBUG",
        est_method="cabaliser",
        graph_state_opt="save",
    )

    input_state = local_simulator.kr(
        local_simulator.zero_state,
        local_simulator.zero_state,
        local_simulator.zero_state,
    )

    with open(cabalize_json0, encoding="utf8") as json_file:
        obj = make_cabalize_dict_compatible(json.load(json_file)[0])
        print(f"Exact simulation of the action of widget0 (dict shown below) on the input_state:\n{obj}")
        input_state = local_simulator.simulate_dict_as_widget(
            obj=obj,
            input_state=input_state,
            table=compiler_tag_table,
        )

    with open(cabalize_json1, encoding="utf8") as json_file:
        obj = make_cabalize_dict_compatible(json.load(json_file)[0])
        print(f"Exact simulation of the action of widget1 (dict shown below) on the input_state:\n{obj}")
        input_state = local_simulator.simulate_dict_as_widget(
            obj=obj,
            input_state=input_state,
            table=compiler_tag_table,
        )

    print("We now apply the inverse quantum ops independent from RRE compilations (ops are reversed)")
    inv_ops = (
        local_simulator.CNOT(n_qubits, 0, 1)
        @ local_simulator.kr(local_simulator.I, local_simulator.I, local_simulator.Tdag)
        @ local_simulator.CZ(n_qubits, 0, 2)
        @ local_simulator.kr(local_simulator.Z, local_simulator.T, local_simulator.S)
        @ local_simulator.CNOT(n_qubits, 0, 1)
        @ local_simulator.kr(local_simulator._Rz(-0.303), local_simulator.Y, local_simulator.X)
        @ local_simulator.SWAP(n_qubits, 2, 1)
        @ local_simulator.kr(local_simulator.Z, local_simulator.H, local_simulator.S)
    )
    input_state = inv_ops @ input_state

    # local_simulator.vec(input_state)  # This prints non-0 elements, useful for debugging
    assert all(abs(value) == 0 for value in input_state[1:])
