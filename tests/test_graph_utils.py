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

"""Unit tests for the `graph_utils` module of rigetti-resource-estimation."""

import networkx as nx
import pytest

from rigetti_resource_estimation.graph_utils import perform_substrate_scheduling


@pytest.mark.parametrize("graph, big_s, big_n", [(nx.path_graph(8), 3, 8), (nx.star_graph(8), 9, 9)])
def test_perform_substrate_scheduling(graph, big_s, big_n):
    """Pytests for perform_substrate_scheduling function."""
    conn_graph, sched = perform_substrate_scheduling(graph)
    assert big_s == len(sched)
    assert big_n == len(conn_graph)
