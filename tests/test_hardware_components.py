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

"""Unite tests for hardware_components.py."""

from numpy import isclose

from rigetti_resource_estimation import Configuration, load_yaml_file
from rigetti_resource_estimation.graph_utils import JabalizerSchedGraphItems
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
widget = widget_table.widgets[0]


def test_quantum_processor():
    """Tests the logic for creation of a quantum processor."""
    processor = QuantumProcessor.from_qubit_number(
        num_qubits=int(1e6),
        pitch_mm=2.0,
        qubits_per_readout_line=10,
        widget=widget,
    )
    assert processor.num_couplers == 20e5
    assert processor.num_junctions == 60e5
    assert processor.num_readout_lines == 1e5
    assert processor.processor_area_sqmm == 40e5


def test_system_exists():
    """Test that a hardware system exists and a method can be run from it."""
    test_system = build_system(widget, config.ftqc_params)
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
        widget=widget,
    )
    test_widget = DistWidget(
        name="test_widget",
        qubits=1200,
        cycles=100,
        p_out=1e-6,
        cond_20to4=False,
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
        num_intermodule_pipes=1,
    )

    def test_calc_num_t_factories(self):
        """Test `calc_num_t_factories` method in FTQC."""
        ftqc = self.ftqc
        assert ftqc.calc_num_t_factories(widget_q=476, d=3, req_logical_qubits=10) == 13

    def test_calc_num_t_factories_returns_0_if_none_fit(self):
        """Test `calc_num_t_factories` method in FTQC returns 0 if no factories fit."""
        ftqc = self.ftqc
        assert ftqc.calc_num_t_factories(widget_q=10000, d=3, req_logical_qubits=10) == 0

    def test_calc_num_t_factories_returns_0_if_logical_qubits_dont_fit(self):
        """Test `calc_num_t_factories` method in FTQC returns 0 the logical qubits won't fit."""
        ftqc = self.ftqc
        assert ftqc.calc_num_t_factories(widget_q=476, d=3, req_logical_qubits=1000) == 0

    def test_ftqc_from_signalchain_nodes(self):
        """Test FTQC's logical and phys attributes constructed from the HW model with the signal components above."""
        test_distance = 1
        ftqc = self.ftqc

        assert ftqc.logical_number_in_rect_comb([100, 2]) == 100
        assert ftqc.logical_number_in_rect_comb([2, 100]) == 100
        assert ftqc.logical_number_in_rect_comb([100, 3]) == 100
        assert ftqc.logical_number_in_rect_comb([100, 8]) == 396
        assert ftqc.logical_number_in_rect_comb([8, 100]) == 396
        assert ftqc.logical_number_in_rect_comb([9, 100]) == 396
        assert ftqc.logical_number_in_rect_comb([50, 30]) == 735
        assert ftqc.logical_number_in_rect_comb([35, 20]) == 340
        assert ftqc.logical_number_in_rect_comb([50, 50]) == 1225

        assert ftqc.allocate_logicals_to_modules(distance=test_distance) == 1225

        test_schedule = [[[1, 2, 3], [5, 6]]]
        measurement_basis_list = [[["T", 3, 0], ["T_Dagger", 2, 0], ["T", 5, 0], ["T", 6, 0], ["RZ", 4, 0]]]
        node_items = JabalizerSchedGraphItems(
            t_count_init=1000,
            rz_count=100,
            clifford_count_init=1000,
            consump_sched=test_schedule,
            prep_sched=test_schedule,
            measurement_basis_list=measurement_basis_list,
        )

        test_schedule_long = [
            [[1, 2, 3, 17, 18, 19], [8, 9, 10], [25, 26, 30]],
            [[1070, 1075, 1080], [2000, 2001, 2002]],
            [[3, 10, 11], [12, 18, 20], [22, 23, 24], [25, 26, 30]],
            [[3, 10, 11], [1066, 1067, 1080], [20, 23, 24], [25, 26, 30]],
            [[1, 2, 3], [5, 6]],
        ]
        measurement_basis_list_long = [
            [["T", 30, 0], ["T_Dagger", 20, 0], ["T", 5, 0], ["T", 8, 0]],
            [["T", 30, 0], ["T_Dagger", 2000, 0], ["T", 5, 0], ["T", 668, 0], ["RZ", 44, 0]],
            [["T_Dagger", 20, 0], ["RZ", 5, 0], ["T", 8, 0]],
            [["T", 30, 0], ["T_Dagger", 2000, 0], ["T", 5, 0], ["T", 668, 0], ["RZ", 44, 0]],
            [["RZ", 5, 0], ["T", 1, 0]],
        ]
        input_nodes_long = [
            [1, 2, 3, 17, 26, 30],
            [1070, 2002],
            [3, 10, 11, 22],
            [3, 10, 11, 1066, 1067, 1080, 20, 23, 24, 25, 26, 30],
            [1, 2, 3],
        ]
        output_nodes_long = [
            [4, 18, 29, 31],
            [1071, 200],
            [15],
            [33, 500, 117, 1166, 1167, 1880, 31],
            [3, 4, 5],
        ]
        node_items_long = JabalizerSchedGraphItems(
            t_count_init=1000,
            rz_count=100,
            clifford_count_init=1000,
            consump_sched=test_schedule_long,
            prep_sched=test_schedule_long,
            measurement_basis_list=measurement_basis_list_long,
            input_nodes=input_nodes_long,
            output_nodes=output_nodes_long,
        )

        consump_sched_times = ftqc.calc_consump_sched_time(
            graph_items=node_items,
            distance=test_distance,
            dist_widget=self.test_widget,
            num_intermodule_pipes=1,
        )
        assert consump_sched_times.intra_consump_ops_time_sec == 16.0
        assert consump_sched_times.intra_t_injection_time_sec == 1616.0
        assert consump_sched_times.inter_handover_ops_time_sec == 0

        consump_sched_times_longer = ftqc.calc_consump_sched_time(
            graph_items=node_items_long,
            distance=test_distance,
            dist_widget=self.test_widget,
            num_intermodule_pipes=2,
        )
        assert consump_sched_times_longer.intra_consump_ops_time_sec == 72.0
        assert consump_sched_times_longer.intra_t_injection_time_sec == 5656.0
        assert consump_sched_times_longer.inter_handover_ops_time_sec == 80.0

        # Hand calculate heat loads (per module)
        #  Line Type           4K per unit  4K heat load   mxc per line     mxc heat load
        # 1000 readout lines    1e-5        2000e-5         2e-7            4000e-7
        # 1000 readout in       1e-5        2000e-5         2e-7            4000e-7
        # 10000 xy lines        3e-5        60000e-5        5e-7             100000e-7
        # 20000 coupler flux    4e-5        160000e-5       6e-7             240000e-7
        # 10000 qflux           4e-5        80000e-5        6e-7             120000e-7
        # sum                               1.52            0.0234

        cryostat_stages = ftqc.calc_thermal_loads()
        assert cryostat_stages[0].temperature_kelvin == 4
        assert cryostat_stages[1].temperature_kelvin == 0.02
        assert isclose(float(cryostat_stages[0].power_dissipation_watt or 0), 1.52 * 2, atol=1e-6)
        assert isclose(float(cryostat_stages[1].power_dissipation_watt or 0), 0.0234 * 2, atol=1e-6)
