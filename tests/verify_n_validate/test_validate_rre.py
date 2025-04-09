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

"""A collection of unit tests to VALIDATE rigetti-resource-estimation operations. Does RRE do the right things?"""

import networkx as nx
import pandas as pd

from rigetti_resource_estimation import load_yaml_file
from rigetti_resource_estimation.estimation_pipeline import estimation_pipeline
from rigetti_resource_estimation.jabalizer_wrapper import create_nxgraph_from_jabalize

PARAMS = load_yaml_file()
QASM_PATH = "./tests/input/qft4.qasm"


def test_validate_estimationpipeline_qft4(tmp_path):
    """
    A validation test for the frontend estimation_pipeline() running a QFT4 algorithm.

    We are comparing relevant outputs from the three internal approaches of `T-counting`, `Graph Processing:
    RubySlippers Compiler` (no padding), and `Graph Processing: Jabalizer Compiler`.
    """
    output_csv = tmp_path / "test.csv"

    # T-couning
    estimation_pipeline(
        circ_path=QASM_PATH,
        log="DEBUG",
        output_csv=output_csv,
        est_method="t_counting",
        graph_state_opt="no_compile",
    )

    # Jabalizer Compiler
    estimation_pipeline(
        circ_path=QASM_PATH,
        log="DEBUG",
        output_csv=output_csv,
        est_method="jabalizer",
        graph_state_opt="save",
    )

    df = pd.read_csv(output_csv, header=0, index_col=None, squeeze=True)  # type: ignore
    results = df.to_dict()

    # Checking if the graphs are the same
    graph_defualt = create_nxgraph_from_jabalize("./tests/input/qft4_all0init_jabalizeframes.json")
    graph_jabalizer = create_nxgraph_from_jabalize("./output/qft4/qft4_all0init_jabalizeframes.json")
    assert nx.is_isomorphic(graph_defualt, graph_jabalizer)

    # Common results expected from T-counting and Jabalizer methods
    assert results["rz_count"][0] == results["rz_count"][1] == 9
    assert results["input_log_qubits"][0] == results["input_log_qubits"][1] == 4
    # Validating results only expected from Jabalizer's approach
    assert results["required_logical_qubits"][1] == 27
    assert results["N"][1] == 30
