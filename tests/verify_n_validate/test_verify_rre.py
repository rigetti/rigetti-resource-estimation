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

"""A collection of unit tests to VERIFY rigetti-resource-estimation operations. Does RRE do things right?"""

import os
import shutil

import pandas as pd

from rigetti_resource_estimation.estimation_pipeline import estimation_pipeline


def test_verify_estimationpipeline_qft4(tmp_path):
    """
    A verification test for the frontend estimation_pipeline() running a widgetized algorithm involving QFT4.

    We start from existing JSONs for the algorithms and find the outputs independent from third-party compilers (using
    "resume" option). We then compare the properties of these small graphs to what we expected to verify the operations
    of RRE.
    """
    output_csv = tmp_path / "test.csv"
    output_json1 = "./output/qft4_random_reps/qft4_random_reps_W1_all0init_cabalizeframes.json"
    output_json2 = "./output/qft4_random_reps/qft4_random_reps_W2_all0init_cabalizeframes.json"

    os.makedirs(os.path.dirname(output_json1), exist_ok=True)
    shutil.copy2(
        "./examples/input/qft4_random_reps_W1_all0init_cabalizeframes.json",
        output_json1,
    )
    os.makedirs(os.path.dirname(output_json2), exist_ok=True)
    shutil.copy2(
        "./examples/input/qft4_random_reps_W2_all0init_cabalizeframes.json",
        output_json2,
    )

    estimation_pipeline(
        log="DEBUG",
        output_csv=output_csv,
        est_method="cabaliser",
        graph_state_opt="resume",
    )

    df = pd.read_csv(output_csv, header=0, index_col=None)
    results = df.to_dict()

    assert results["input_logical_qubits"][0] == 4
    assert results["N"][0] == 216
    assert results["num_logical_qubits_per_busrail"][0] == 22
