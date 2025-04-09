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

"""A collection of unit tests to VERIFY rigetti-resource-estimation operations. Does RRE do things right?"""

import os
import shutil
import math

import pandas as pd
from numpy import isclose

from rigetti_resource_estimation.estimation_pipeline import estimation_pipeline


def test_verify_estimationpipeline_qft4(tmp_path):
    """
    A verification test for the frontend estimation_pipeline() running not-exactly-QFT3 algorithm (a circuit inspired by
    QFT3 but simpler, where the graph can be simply worked out).

    We start from an _existing_ not-exactly-QFT3 .json graph and find the outputs independent from TPLs. We then
    compare the properties of this small graph to what is expected to verify the operations of RRE.
    """
    circ_path = "./tests/input/qft4.qasm"
    output_csv = tmp_path / "test.csv"

    os.makedirs(os.path.dirname("./output/qft4/qft4_all0init_jabalizeframes.json"), exist_ok=True)
    shutil.copy2(
        "./tests/input/qft4_all0init_jabalizeframes.json",
        "./output/qft4/qft4_all0init_jabalizeframes.json",
    )

    estimation_pipeline(
        circ_path=circ_path,
        log="DEBUG",
        output_csv=output_csv,
        est_method="jabalizer",
        graph_state_opt="resume",
    )

    df = pd.read_csv(output_csv, header=0, index_col=None, squeeze=True)  # type: ignore
    results = df.to_dict()

    assert results["rz_count"][0] == 9
    assert results["input_log_qubits"][0] == 4
    assert results["N"][0] == 30
    assert results["required_logical_qubits"][0] == 27


def test_verify_estimationpipeline_decomposed_qft10(tmp_path):
    """Verify the estimation_pipeline() using a widgetized max-depth-3 QFT10 with building block QASMs."""
    # A single full QFT10 algorithm
    estimation_pipeline(
        circ_path="./tests/input/qft10.qasm",
        log="DEBUG",
        output_csv=tmp_path / "test2.csv",
        est_method="jabalizer",
        graph_state_opt="save",
    )

    # A decomposed QFT10 JSON with the max depth of 3
    estimation_pipeline(
        circ_path="./tests/input/qft10-decomposed3.json",
        log="DEBUG",
        decomposed_bb=True,
        output_csv=tmp_path / "test3.csv",
        est_method="jabalizer",
        graph_state_opt="save",
    )

    # csvs have different number of columns, so they must be loaded separately and then concatenated
    df1 = pd.read_csv(tmp_path / "test2.csv", header=0, index_col=None, squeeze=True)  # type: ignore
    df2 = pd.read_csv(tmp_path / "test2.csv", header=0, index_col=None, squeeze=True)  # type: ignore
    df = pd.concat([df1, df2], axis=0).reset_index()
    results = df.to_dict()

    # logical-circuit-level properties
    assert results["rz_count"][0] == results["rz_count"][1] == 108
    assert results["input_log_qubits"][0] == results["input_log_qubits"][1] == 10
    # compiled-level properties
    assert isclose(math.log10(results["t_count"][0]), math.log10(results["t_count"][1]), atol=1)
    assert isclose(math.log10(results["distance"][0]), math.log10(results["distance"][1]), atol=1)
    assert isclose(
        math.log10(results["required_logical_qubits"][0]), math.log10(results["required_logical_qubits"][1]), atol=1
    )
    assert isclose(math.log10(results["S_consump"][0]), math.log10(results["S_consump"][1]), atol=1)
    assert isclose(math.log10(results["N"][0]), math.log10(results["N"][1]), atol=1)
