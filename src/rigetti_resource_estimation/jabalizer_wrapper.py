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
**Module** ``jabalizer_wrapper``

A set of utility and wrapper functions to call and extend methods from the Jabalizer (originally in Julia) and Pauli
Tracker (originally in Rust) libraries.

Parts of the following code are inspired and/or customized from the original templates in the open-source references of:
[1] https://github.com/zapatacomputing/benchq
[2] https://github.com/taeruh/mbqc_scheduling
"""

from os import path
import pathlib
import logging

import json
from juliacall import Main as jl  # pylint: disable=no-name-in-module  # type: ignore
import networkx as nx


jl.include(
    path.join(pathlib.Path(__file__).parent.resolve(), "jabalizer_wrapper.jl"),
)

logger = logging.getLogger(__name__)


def get_algorithmic_graphdata_from_jabalizer(
    circuit: str,
    circuit_fname: str,
    suffix: str,
    pcorrections_flag: bool = False,
) -> str:
    """
    Create spatial graph, Pauli frames, consumption schedule, and mdata from `circuit` using the Jabalizer compiler.

    :param circuit: a logical-level circuit in QASM str format.
    :param circuit_fname: the stem for the logical circuit file path.
    :param suffix: a suffix str to add to the filename of jabalize JSONs in the form of
        circuit_fname + suffix + "_all0init_jabalize.json".

    :returns: path to the jabalize JSON file generated by Jabalizer containing the spacial graph, pauli frames and
        some metadata.
    """
    debug_flag = True if logging.root.level > 0 else False

    jl.run_jabalizer_mbqccompile(circuit, circuit_fname, suffix, pcorrections_flag, debug_flag)

    jabalize_json_path = "output/" + circuit_fname + "/" + circuit_fname + suffix + "_all0init_jabalizeframes.json"
    if not path.isfile(jabalize_json_path):
        raise RuntimeError("Jabalizer could not create jabalize JSON file.")

    return jabalize_json_path


def create_nxgraph_from_jabalize(jabalize_json_path: str) -> nx.Graph:
    """Create a NetworkX graph from a jabalize node list.

    :param jabalize_json_path: path to file containing json file output from Jabalizer.
    """
    with open(jabalize_json_path, encoding="utf8") as f:
        jabalize_data = json.load(f)
    graph_list = jabalize_data["spatialgraph"]

    graph = nx.Graph()
    graph.add_nodes_from(list(range(len(graph_list))))
    for i, node_list in enumerate(graph_list[:-1]):
        for j in node_list:
            if j > i:
                graph.add_edge(i, j)
    return graph


def get_mbqc_data_from_jabalizejson(mbqc_output_qasm: str, jabalize_json: str) -> None:
    """Run Julia's Jabalizer.qasm_instruction to compile a MBQC-formatted qasm based on `jabalize_json` input.

    :param mbqc_output_qasm: MBQC's output QASM string.
    :param jabalize_json: input json file for Jabilizer
    """
    logger.info("Employing Jabalizer to compile a MBQC-formatted qasm based on `jabalize_json` input ...")
    jl.get_mbqc_data_from_jabalizejson(mbqc_output_qasm, jabalize_json)