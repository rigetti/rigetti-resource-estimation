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

"""Unit tests for `rigetti_resource_estimation` cabaliser_wrapper module."""
import numpy as np
from rigetti_resource_estimation.cabaliser_wrapper import CabaliserCompiler


def test_measurement_basis_list_is_not_all_zero():
    """Test to ensure the CabaliserCompiler doesn't return all zeros for the measurement tags.

    Issue 102. https://github.com/rigetti/rigetti-resource-estimation-dev/issues/102
    """
    comp = CabaliserCompiler("bug", "save")

    # arbitrary, with some non-zero angle tags
    transpiled_widget = [
        (36, (0,)),
        (128, (1, 31)),
        (32, (0,)),
        (128, (3, 10)),
        (32, (0,)),
        (32, (0,)),
        (128, (2, 31)),
        (32, (0,)),
        (128, (0, 1234)),
        (128, (0, 333)),
    ]
    num_qubits = 4
    max_memory = 10
    result = comp.compile(transpiled_widget, num_qubits, max_memory)
    assert not np.isclose(sum(abs(np.array(result["measurement_tags"]))), 0), "All measurement bases are 0."
