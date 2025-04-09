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

"""Unit tests for the estimate_from_surgery module of rigettiestimator."""

from rigetti_resource_estimation.estimate_from_graph import d_from_rates
from rigetti_resource_estimation.graph_utils import JabalizerSchedGraphItems
import rigetti_resource_estimation.hardware_components as hw
from rigetti_resource_estimation import Configuration, load_yaml_file


# Initialize some params and objects for the unit tests
config = Configuration(load_yaml_file())
widget_table = hw.build_dist_widgets(config.widget_params)
widget = widget_table.widgets[0]

prep_sched = [
    [(1, (2, 3)), (17, (18, 19)), (8, (9, 10)), (25, (26, 30))],
    [(3, (10, 11)), (12, (18, 20)), (22, (23, 24)), (25, (26, 30))],
    [(3, (10, 11)), (12, (18, 20)), (20, (23, 24)), (25, (26, 30))],
]
consump_sched = [
    [[7, 18, 19], [21, 22]],
    [[9, 10, 11], [13, 18, 21], [20, 23, 24, 28, 26, 30]],
    [[19, 20]],
]
node_items = JabalizerSchedGraphItems(
    t_count_init=1000,
    rz_count=100,
    clifford_count_init=1000,
    big_n=12,
    delta=4,
    t_length_unit=50,
    prep_sched=prep_sched,
    consump_sched=consump_sched,
)


def test_d_from_rates():
    """Test plogcell_from_palg static method in Qentiana."""
    assert 12 == d_from_rates(
        widget=widget,
        p_algo=0.5,
        graph_info=node_items,
        p_gates=config.p_gates,
        k=config.error_scaling_coeffs["k"],
        p_th=config.error_scaling_coeffs["p_th"],
    )
