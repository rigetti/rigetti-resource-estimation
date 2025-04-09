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

"""Unite tests for hardware_components.py."""

from numpy import isclose

from rigetti_resource_estimation import Configuration, load_yaml_file
from rigetti_resource_estimation.graph_utils import CompilerSchedGraphItems
from rigetti_resource_estimation.hardware_components import (
    Amplifier,
    Attenuator,
    build_dist_widgets,
    build_system,
    Cable,
    Circulator,
    CryostatStage,
    Fridge,
    FTQC,
    LineType,
    QuantumProcessor,
    SignalChain,
    DistWidget,
)

config = Configuration(load_yaml_file())
widget_table = build_dist_widgets(config.widget_params)
distill_widget = widget_table.widgets[0]
widgetized_circ_dict = {
    "stitches": {("W1", "W1"): 7, ("W1", "W2"): 2, ("W2", "W1"): 2},
    "widgets": {"W1": (None, 10), "W2": (None, 2)},
    "first_widget": "W1",
    "compiler_tag_table": {},
    "input_qubits": 26,
    "init_t_count": None,
    "init_rz_count": None,
    "init_clifford_count": None,
}


def test_quantum_processor():
    """Tests the logic for creation of a quantum processor."""
    processor = QuantumProcessor.from_qubit_number(
        num_qubits=int(1e6),
        pitch_mm=2.0,
        qubits_per_readout_line=10,
        widget=distill_widget,
    )
    assert processor.num_couplers == 20e5
    assert processor.num_junctions == 60e5
    assert processor.num_readout_lines == 1e5
    assert processor.processor_area_sqmm == 40e5


def test_system_exists():
    """Test that a hardware system exists and a method can be run from it."""
    test_system = build_system(distill_widget, config.ftqc_params)
    thermal_loads = test_system.calc_thermal_loads()
    assert isclose(thermal_loads[1].temperature_kelvin, 0.02, rtol=1e-5)
    assert isclose(float(thermal_loads[1].power_dissipation_watt or 0), 0.000168, rtol=1e-5)
    assert isclose(thermal_loads[0].temperature_kelvin, 4.0, rtol=1e-5)
    assert isclose(float(thermal_loads[0].power_dissipation_watt or 0), 840.0, rtol=1e-5)


class TestFTQC:
    """ "A class to test FTQC construction and attributes."""

    coax_cable = Cable(name="coaxial_cable", length_m=1.2)
    atten_20 = Attenuator(name="20dB attenuator", att_decibel=20)
    atten_40 = Attenuator(name="20dB attenuator", att_decibel=40)
    atten_5 = Attenuator(name="5dB attenuator", att_decibel=5)
    circ = Circulator(name="circulator", isolation_decibel=18, loss_decibel=2)
    hemt = Amplifier(name="HEMT amplifier", gain_decibel=40, noise_kelvin=2.1)
    fridge_stage_4kelvin = CryostatStage(temperature_kelvin=4, cooling_power_watt=1e-3)
    fridge_stage_mxc = CryostatStage(temperature_kelvin=0.02, cooling_power_watt=1e-5)

    ro_stage_4k = CryostatStage(temperature_kelvin=4, component=atten_5, power_dissipation_watt=1e-5)
    ro_stage_mxc = CryostatStage(temperature_kelvin=0.02, component=atten_20, power_dissipation_watt=2e-7)
    xy_stage_4k = CryostatStage(temperature_kelvin=4, component=atten_40, power_dissipation_watt=3e-5)
    xy_stage_mxc = CryostatStage(temperature_kelvin=0.02, component=atten_20, power_dissipation_watt=5e-7)
    flux_stage_4k = CryostatStage(temperature_kelvin=4, component=atten_5, power_dissipation_watt=4e-5)
    flux_stage_mxc = CryostatStage(temperature_kelvin=0.02, component=atten_20, power_dissipation_watt=6e-7)

    ro_in_list = [coax_cable, coax_cable, ro_stage_4k, ro_stage_mxc]
    ro_out_list = [
        coax_cable,
        circ,
        circ,
        hemt,
        ro_stage_4k,
        coax_cable,
        coax_cable,
        ro_stage_mxc,
    ]
    xy_list = [coax_cable, coax_cable, xy_stage_4k, xy_stage_mxc]
    flux_list = [coax_cable, coax_cable, flux_stage_4k, coax_cable, flux_stage_mxc]
    readout_line_in = SignalChain(
        name="readout input",
        component_current=0.1,
        line_type=LineType.READOUT_IN,
        components=ro_in_list,
    )
    readout_line_out = SignalChain(
        name="readout output",
        component_current=0.1,
        line_type=LineType.READOUT_OUT,
        components=ro_out_list,
    )
    xy_line = SignalChain(
        name="xy lines",
        component_current=0.01,
        line_type=LineType.MICROWAVE_XY,
        components=xy_list,
    )
    qubit_flux_line = SignalChain(
        name="qubit flux line",
        component_current=1.3,
        line_type=LineType.QUBIT_FLUX,
        components=flux_list,
    )
    coupler_flux_line = SignalChain(
        name="coupler flux line",
        component_current=2.3,
        line_type=LineType.COUPLER_FLUX,
        components=flux_list,
    )
    fridge = Fridge(name="blueFors XLD", stages=[fridge_stage_4kelvin, fridge_stage_mxc])
    processor = QuantumProcessor.from_qubit_number(
        num_qubits=10000,
        pitch_mm=2.0,
        qubits_per_readout_line=10,
        widget=distill_widget,
    )
    test_dist_widget = DistWidget(
        name="test_dist_widget",
        qubits=1200,
        cycles=100,
        p_out=1e-6,
        cond_20to4=False,
        length=31,
        width=45,
    )

    ftqc = FTQC(
        name="foo_system",
        cryostat=fridge,
        qpu=processor,
        xy_line=xy_line,
        qubit_flux_line=qubit_flux_line,
        coupler_flux_line=coupler_flux_line,
        readout_out_line=readout_line_out,
        readout_in_line=readout_line_in,
        inter_handover_timescale_sec=10.0,
        intra_qcycle_sec=1.0,
        num_pipes_per_intermodule_connection=1,
    )

    def test_calc_num_t_factories(self):
        """Test `calc_num_t_factories` method in FTQC."""
        intra_component_counts = self.ftqc.calc_num_t_factories(dist_widget=self.test_dist_widget, d=3, delta=10)
        assert intra_component_counts.num_t_factories == 4
        assert intra_component_counts.edge_logical == 23
        assert intra_component_counts.len_transfer_bus == 108

    def test_ftqc_from_signalchain_nodes(self):
        """Test FTQC's logical and phys attributes constructed from the HW model with the signal components above."""
        test_distance = 1
        num_modules_per_leg = 2

        test_consump_schedule = [
            [[{9: [7, 8]}, {13: [8]}, {16: [12]}], [{15: [12, 9]}, {14: [13]}, {17: [16]}]],
            [[{0: []}, {2: []}, {3: []}, {1: []}]],
        ]
        test_prep_schedule = [[[(3, (3, 7))], [(4, (0, 4)), (9, (7, 9))]], [[(25, (0, 25))]]]
        measurement_basis_list = [[0, 0], [0, 0, 0]]
        input_nodes = [[0, 7], [12, 25]]
        output_nodes = [[1, 24], [3, 8]]
        node_items = CompilerSchedGraphItems(
            t_count_init=10000,
            rz_count=1000,
            clifford_count_init=5000,
            consump_sched=test_consump_schedule,
            prep_sched=test_prep_schedule,
            measurement_basis_list=measurement_basis_list,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            delta=2000,
        )
        consump_sched_times = self.ftqc.calc_consump_sched_times(
            graph_items=node_items,
            distance=test_distance,
            dist_widget=self.test_dist_widget,
            num_pipes_per_intermodule_connection=1,
            transpiled_widgets=widgetized_circ_dict,
        )

        assert (
            self.ftqc.allocate_logicals_to_modules(
                d=test_distance, dist_widget=self.test_dist_widget, delta=int(node_items.delta or 0)
            )
            == num_modules_per_leg
        )
        assert consump_sched_times.intra_consump_ops_time_sec == 176.0
        assert consump_sched_times.intra_distill_delay_sec == 0
        assert consump_sched_times.intra_prep_delay_sec == 176.0
        assert consump_sched_times.inter_handover_ops_time_sec == 1760.0

        # Hand calculations for heat loads (per module)
        #  Line Type           4K per unit  4K heat load   mxc per line     mxc heat load
        # 1000 readout lines    1e-5        2000e-5         2e-7            4000e-7
        # 1000 readout in       1e-5        2000e-5         2e-7            4000e-7
        # 10000 xy lines        3e-5        60000e-5        5e-7             100000e-7
        # 20000 coupler flux    4e-5        160000e-5       6e-7             240000e-7
        # 10000 qflux           4e-5        80000e-5        6e-7             120000e-7
        # sum                               1.52            0.0234

        cryostat_stages = self.ftqc.calc_thermal_loads(num_modules_per_leg)
        assert cryostat_stages[0].temperature_kelvin == 4
        assert cryostat_stages[1].temperature_kelvin == 0.02
        assert isclose(float(cryostat_stages[0].power_dissipation_watt or 0), 1.52 * 2 * num_modules_per_leg, atol=1e-6)
        assert isclose(
            float(cryostat_stages[1].power_dissipation_watt or 0), 0.0234 * 2 * num_modules_per_leg, atol=1e-6
        )
