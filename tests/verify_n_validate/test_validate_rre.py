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

"""A collection of unit tests to VALIDATE rigetti-resource-estimation operations. Does RRE do the right things?"""

import json
import networkx as nx
import pandas as pd

from rigetti_resource_estimation import load_yaml_file
from rigetti_resource_estimation.estimation_pipeline import estimation_pipeline
from rigetti_resource_estimation.cabaliser_wrapper import create_nxgraph_from_adjdict


PARAMS = load_yaml_file()


def test_validate_estimationpipeline_qft4(tmp_path):
    """
    A validation test for the frontend estimation_pipeline() running a  widgetized algorithm involving QFT4.

    We compare relevant outputs from the two internal approaches of `T-counting` and `Cabaliser` compilation.
    """
    output_csv = tmp_path / "test.csv"

    # T-couning
    estimation_pipeline(
        log="DEBUG",
        output_csv=output_csv,
        est_method="t_counting",
        graph_state_opt="no_compile",
    )

    # Cabaliser Compiler
    estimation_pipeline(
        log="DEBUG",
        output_csv=output_csv,
        est_method="cabaliser",
        graph_state_opt="save",
    )

    # Checking if the generated graph W2 and the expected one are equivalent
    json_defualt = "examples/input/qft4_random_reps_W2_all0init_cabalizeframes.json"
    with open(json_defualt, encoding="utf8") as json_file:
        data_default = json.load(json_file)[0]
    json_cabaliser = "output/qft4_random_reps/qft4_random_reps_W2_all0init_cabalizeframes.json"
    with open(json_cabaliser, encoding="utf8") as json_file:
        data_cabaliser = json.load(json_file)[0]
    graph_default = create_nxgraph_from_adjdict(data_default["adjacencies"])
    graph_cabaliser = create_nxgraph_from_adjdict(data_cabaliser["adjacencies"])
    assert nx.is_isomorphic(graph_default, graph_cabaliser)

    df = pd.read_csv(output_csv, header=0, index_col=None)
    results = df.to_dict()
    # Checking common results expected from T-counting and Cabaliser methods
    assert results["input_logical_qubits"][0] == results["input_logical_qubits"][1] == 4
    # Validating results only expected from Cabaliser's approach
    assert results["num_logical_qubits_per_busrail"][1] == 22
    assert results["N"][1] == 216
