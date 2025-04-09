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
**Module** ``cabaliser_wrapper``

A set of utility and wrapper functions to call and extend methods from the Cabaliser (originally in C) and Pauli
Tracker (originally in Rust) libraries.

Parts of the following code are inspired and/or customized from the original templates in the open-source references of:
[1] https://github.com/zapatacomputing/benchq
[2] https://github.com/taeruh/mbqc_scheduling
"""

from pathlib import Path
import logging
from typing import Literal

import json
import networkx as nx

from cabaliser.operation_sequence import OperationSequence
from cabaliser.widget_sequence import WidgetSequence

logger = logging.getLogger(__name__)


class CabaliserCompiler:
    """Compiler class for Cabaliser."""

    def __init__(
        self,
        circuit_fname: str,
        graph_state_opt: Literal["no_compile", "save", "resume"],
    ):
        """Initialize some graph state parameters based on user inputs.

        :param circuit_fname: the stem for the logical circuit file path.
        :param graph_state_opt: What to do with respect to the graph state compilation. Options are explained below.
            'no_compile': The graph compilation pipeline will NOT be executed (relevant for, e.g., for 't_counting'
            approaches).
            'save': RRE compiles the subcircuit and attempt to generate the graph states and Pauli Frames info through
            Cabaliser compiler. The program will save all outputs in JSON file in the subdirectory
            `output/<circuit_fname>/`. The outputs will be named `<circuit_fname>_all0init_cabalizeframes.json`.
            'resume': RRE will try to resume the calculations assuming an `<circuit_fname>_all0init_cabalizeframes.json`
            already exist in `output/<circuit_fname>/` subdirectory.
        """
        self.graph_state_opt = graph_state_opt
        self.circuit_fname = circuit_fname

    def compile(self, transpiled_widget: list, input_qubits: int, max_memory_qubits: int, widget_suffix: str = ""):
        "Compile a `transpiled_widget` and save the outputted JSON if requested."
        widget = self._compile_widget2json(transpiled_widget, input_qubits, max_memory_qubits)
        compiled_widget_json = widget.json()
        del widget
        cabalise_json_path = Path(
            "output/" + self.circuit_fname + "/" + self.circuit_fname + widget_suffix + "_all0init_cabalizeframes.json"
        )
        if self.graph_state_opt == "save":
            cabalise_json_path.parent.mkdir(exist_ok=True, parents=True)
            with open(cabalise_json_path, "w", encoding="utf8") as f:
                json.dump(compiled_widget_json, f)
        return compiled_widget_json[0]

    def _compile_widget2json(self, transpiled_widget: list, input_qubits: int, max_memory_qubits: int):
        """
        Compile `transpiled_widget` to create a Cabaliser's widget object representing graph and scheduling data.

        :param transpiled_widget: a list of transpiled logical-level widgets in Cabaliser-readable format.
        :param input_qubits: the number of input qubits for `transpiled_widget`.
        :param max_memory_qubits: An upper bound to the maximum quantum memory size assigned to the widgets for
            Cabaliser's compilation.
        """
        ops = OperationSequence(len(transpiled_widget))
        for opcode, args in transpiled_widget:
            ops.append(opcode, *args)

        logger.info("Cabaliser is creating widget through sequencing and stitching sub-widgets\n")
        widget = WidgetSequence(input_qubits, max_memory_qubits)

        logger.info("Cabaliser is applying gate operations on the widget\n")
        widget(ops, store_output=True, rz_to_float=False)

        return widget


def create_nxgraph_from_adjdict(adjacency_data: dict) -> nx.Graph:
    """Create a NetworkX graph from a adjacency matrix written as a dict.

    :param adjacency_data: A adjacency matrix written as a dict.
    """
    graph_list = list(adjacency_data.values())

    graph = nx.Graph()
    graph.add_nodes_from(list(range(len(graph_list))))
    for i, node_list in enumerate(graph_list[:-1]):
        for j in node_list:
            if j > i:
                graph.add_edge(i, j)
    return graph
