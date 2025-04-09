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
**Module** ``rigetti_resource_estimation.resources``

This module defines Resource classes to hold information and methods related to resources estimated through RRE.
"""

import sys
from abc import abstractmethod
from typing import Dict, Union, Type, Optional, Literal, Sequence, get_type_hints
import math
import logging

import pandas as pd

from rigetti_resource_estimation.estimate_from_graph import Experiment

logger = logging.getLogger(__name__)

THIS_MODULE = sys.modules[__name__]
ResourceValue = Union[float, int]
ResourceFormat = Union[Type[int], Type[float], Type[str]]


def find_physq_perpatch(distance: int) -> int:
    """Find how many data and syndrome qubits are necessary per logical patches for a given distance.

    Formula is based on rotated surface code patches.

    :param distance: surface code distance

    :returns: physical qubits per patch
    """
    return 2 * (distance**2)


class Resource:
    """Base class for all Resource subclasses."""

    def __init__(self, name: str, short_name: str, description: str, unit: Optional[str] = None) -> None:
        """
        :param name: full name of the resource.
        :param short_name: name used in CSV reports, or when using the full name would be cumbersome.
        :param description: human readable text describing the resource.
        :param unit: unit the resource is reported in.
        """
        self.name = name
        self.short_name = short_name
        self.description = description
        self.unit = unit

    @abstractmethod
    def get_value(self, experiment: Experiment) -> ResourceValue:
        """Return calculated value of this resource."""

    @property
    def fmt(self):
        """Return the type of the return of get_value. Used for formatting tables."""
        return get_type_hints(self.get_value)["return"]


class Distance(Resource):
    """Surface code distance."""

    def __init__(
        self, name: str = "Distance", short_name: str = "distance", description: str = "Surface code distance, d"
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated value of this resource.

        :param experiment: experiment to calculate the distance for.
        """
        return experiment.distance


class RequiredLogicalQubits(Resource):
    """
    Resource class for the upper bound on the number of logical qubits (max memory) required at any given time.

    We report the logical patches required per fridge and per each rail of bilinear quantum. We ignore
    the patches required for the T-distillation widgets. This resource is the main identifier for the
    space costs.
    """

    def __init__(
        self,
        name: str = "Required Logical Qubits",
        short_name: str = "required_logical_qubits",
        description: str = "Required number of logical qubits, max memory or graph delta",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated required number of logical qubits.

        :param experiment: experiment to calculate the required number of logical qubits for.
        """
        # logger.debug(f"The value of {delta} was assigned to resource class `Delta`.")
        return experiment.graph_items.delta or 0


class DistillWidgetQubits(Resource):
    """Number of qubits in the distillation widget (T-factory)."""

    def __init__(
        self,
        name: str = "t_widget_qb",
        short_name: str = "wq",
        description: str = "Physical qubits per T-widget",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated value of Widget Qubits.

        :param experiment: experiment to calculate the widget qubits value for.
        """
        return experiment.widget.qubits


class NumTFactories(Resource):
    """Total number of T-factories in both fridges."""

    def __init__(
        self,
        name: str = "Number of T Factories",
        short_name: str = "num_t_factories",
        description: str = "Number of T-factories in the distillation fridge(s)",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return total number of T-factories present in both fridges.

        :param experiment: experiment to calculate the total T factories for.
        """
        return 2 * experiment.num_t_factories


class ReqQubitsTFactories(Resource):
    """Calculate the total number of physical qubits required for T-factory widgets per fridge."""

    def __init__(
        self,
        name: str = "Required physical qubits for T-factories",
        short_name: str = "req_phys_qubits_T_factories",
        description: str = "Required physical qubits for T-factories",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated value of physical qubits used for T-factories.


        :param experiment: experiment to calculate the total number of T-factory qubits for.
        """
        return experiment.num_t_factories * experiment.widget.qubits


class ReqPhysicalQubits(Resource):
    """
    Calculate the total number of physical qubits required in the fault-tolerant computer at any given time.

    We assume a bilinear-bus two fridge architecture for the complete FTQC by default:
    The architecture includes two interconnected fridge modules hosting algorithmic time-ordered subcircuit widgets and
    T-factories. The two modules run interleaving subgraphs. When a module grows the graph by preparing a new subgraph
    inside, the other in contrast consumes the prior subgraph, which is assumed to always take longer than any graph
    preparation. The quantum bus inside both modules is bilinear and ancilla qubits are laid out in a comb-like pattern.
    Both module host an equal number of T-factory widgets inside, which feed T states internally to the fridge that is
    consuming the subgraph.

    Users may switch to a single-fridge architecture (with T-factories inside) by appropriately configuring
    `inter_handover_timescale_sec`, `qcycle_char_timescale_ns`, and `processor.num_qubits` in `params.yaml`. One can
    also set `num_intermodule_pipes` to speed-up subgraph handover to any desired number.
    """

    def __init__(
        self,
        name: str = "Total required physical qubits",
        short_name: str = "total_req_physical_qubits",
        description: str = "Total required physical qubits",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated value of physical qubits.


        :param experiment: experiment to calculate the total number of qubits for.
        """
        delta_val = experiment.graph_items.delta or 0
        return (
            2 * delta_val * find_physq_perpatch(experiment.distance)
            + experiment.num_t_factories * experiment.widget.qubits
        )


class InputLogicalQubits(Resource):
    """Number of logical qubits in the input circuit."""

    def __init__(
        self,
        name: str = "Input Logical Qubits",
        short_name: str = "input_log_qubits",
        description: str = "Number of logical qubits in the input algorithm",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of logical qubits in the input circuit.

        :param experiment: experiment to calculate the number of logical qubits for.
        """
        return experiment.input_log_qubits


class DiamondNormEps(Resource):
    """Diamond norm eps value for gate-synth ops to decompose arbitrary Rz's to Clifford+T."""

    def __init__(
        self,
        name: str = "Diamond norm eps",
        short_name: str = "diamond_norm_eps",
        description: str = "Diamond-norm epsilon for gate-synth decompositions, eps",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> float:
        """Return the diamond norm eps value.

        :param experiment: experiment to calculate the diamond norm for.
        """
        return experiment.diamond_norm_eps


class TCount(Resource):
    """Input circuit's total T-count when one performs all the required gate-synth ops."""

    def __init__(
        self,
        name: str = "T count",
        short_name: str = "t_count",
        description: str = "Total number of T-basis measurements required at the distill-consump stage, T-count",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the circuit's T-Count.

        :param experiment: experiment to calculate the T-Count for.
        """
        return experiment.graph_items.t_count()


class TDepth(Resource):
    """Estimated T-depth for the input logical circuit."""

    def __init__(
        self, name: str = "Circuit T-depth", short_name: str = "t_depth", description: str = "Circuit T-depth"
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the circuit's T depth.

        :param experiment: experiment to calculate the T depth for.
        """
        node_items = experiment.graph_items
        if node_items.delta in [None, 0]:
            raise RuntimeError(f"Captured delta={node_items.delta}, making TDepth calculations impossible.")
        else:
            delta = node_items.delta or 0
        t_depth = math.ceil(node_items.t_count() / delta)
        return t_depth


class InitRZCount(Resource):
    """Initial non-Clifford-angle RZ count for the input circuit."""

    def __init__(
        self,
        name: str = "RZ Count",
        short_name: str = "rz_count",
        description: str = "Number of Rz unitaries with non-Clifford angles in the initial logical circuit, Rz-count",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the circuit's RZ count.

        :param experiment: experiment to calculate the RZ count for.
        """
        return experiment.graph_items.rz_count or 0


class InitTCount(Resource):
    """Number of initial logical T gates for the input circuit."""

    def __init__(
        self,
        name: str = "Initial T Count",
        short_name: str = "init_T_count",
        description: str = """"Number of the T gates in the initial logical circuit prior to performing Clifford+T
            decompositions.""",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the circuit's initial T count.

        :param experiment: experiment to calculate the initial T count for.
        """
        return experiment.graph_items.t_count_init or 0


class InitCliffordCount(Resource):
    """Number of explicit logical Clifford gates for the input circuit."""

    def __init__(
        self,
        name: str = "Initial Clifford Count",
        short_name: str = "init_clifford_count",
        description: str = "Number of the explicit Clifford gates in the initial logical circuit",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the circuit's initial Clifford count.

        :param experiment: experiment to calculate the initial Clifford count for.
        """
        return experiment.graph_items.clifford_count_init or 0


class GraphN(Resource):
    """N-value (size) of the complete graph state."""

    def __init__(self, name: str = "N", short_name: str = "N", description: str = "Graph nodes, N") -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the graph's N value.

        :param experiment: experiment to calculate the graph's N value for.
        """
        return experiment.graph_items.big_n or 0


class ConsumpScheduleSteps(Resource):
    """Total number of measurement steps for graph consumption scheduling, S_consump."""

    def __init__(
        self,
        name: str = "S_consump",
        short_name: str = "S_consump",
        description: str = "Graph S for consump schedule",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the graph's schedule steps.

        :param experiment: experiment to calculate the graphs schedule for.
        """
        return experiment.graph_items.big_s_consump()


class PrepScheduleSteps(Resource):
    """Total number of measurement steps for graph initialization scheduling, S_prep."""

    def __init__(
        self,
        name: str = "S_prep",
        short_name: str = "S_prep",
        description: str = "Graph S for prep schedule",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the graph's preparation schedule steps.

        :param experiment: experiment to calculate the graph preparation schedule steps for.
        """
        return experiment.graph_items.big_s_prep()


class MaxIONodes(Resource):
    """
    Maximum number of output, or equivalently input, nodes in the output_nodes and input_nodes of graph_items.

    For the time-sliced widgetization we perform on the input algorithm, all widgets have the same output and input
    nodes, equal to this reported number.
    """

    def __init__(
        self,
        name: str = "max_IO_nodes",
        short_name: str = "max_IO_nodes",
        description: str = "Maximum number of input/output nodes",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the maximum IO nodes.

        :param experiment: experiment to calculate the maximum IO nodes for.
        """
        output_nodes = experiment.graph_items.output_nodes if experiment.graph_items.output_nodes else [[]]
        return max([len(subset) for subset in output_nodes])


class DecoderTock(Resource):
    """Decoder tock (seconds)."""

    def __init__(
        self,
        name: str = "Decoding tock",
        short_name: str = "decoder_tock",
        description: str = "Decoder tock in seconds",
        unit="second",
    ) -> None:
        super().__init__(name, short_name, description, unit)

    def get_value(self, experiment: Experiment) -> float:
        """Return the decoder tock time.

        Based on the first equation of Sec. II.B. of [Chubb 2021, arXiv:2101.04125]. Note a single-core unit is
        responsible for d-sweeps, no parallelization or staggered architecture is considered here.

        :param experiment: experiment to calculate the decoder tock time for.
        """
        nnn = 2 * find_physq_perpatch(experiment.distance) - 1  # number of decoding tensors in a logical patch
        nnn0 = 2 * find_physq_perpatch(96) - 1  # number of decoding tensors in a logical patch (reference machine)
        bond_dim0 = 20
        bond_dim = experiment.bond_dim
        time_scale = experiment.decoder_char_timescale_sec
        numerator = nnn * (math.log(nnn) + math.pow(bond_dim, 3))
        denom = nnn0 * (math.log(nnn0) + math.pow(bond_dim0, 3))
        decoder_patch_time_sec = (numerator / denom) * time_scale
        return experiment.distance * decoder_patch_time_sec


class QuantumIntraTock(Resource):
    """Intra-modular tock (seconds) for surface code operations and parity measurements at consumption stage."""

    def __init__(
        self,
        name: str = "Quantum Tock",
        short_name: str = "quantum_tock",
        description: str = "Quantum intra tock for consumption stage (seconds)",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> float:
        """Return the quantum tock time.

        Note a single unit is responsible for d-sweeps, no parallelization or staggered arch is considered.

        :param experiment: experiment to calculate the quantum tock time for.
        """
        return 8 * experiment.distance * experiment.system_architecture.intra_qcycle_sec


class TStateTock(Resource):
    """Intra-modular tock (seconds) to both distill and inject a purified T-state for graph consumption."""

    def __init__(
        self,
        name: str = "T-state Distill-transfer Tock",
        short_name: str = "t_state_tock",
        description: str = "Intra-module tock for T-state injections and transfers (seconds)",
        unit: str = "second",
    ) -> None:
        super().__init__(name, short_name, description, unit)

    def get_value(self, experiment: Experiment) -> float:
        """Return the T state tock time.

        :param experiment: experiment to calculate the T state tock time for.
        """
        return 8 * (experiment.distance + experiment.widget.cycles) * experiment.system_architecture.intra_qcycle_sec


class AvailPhysicalQubits(Resource):
    """Total number of physical qubits available in the both modules of the FTQC"""

    def __init__(
        self,
        name: str = "Total number of available physical qubits",
        short_name: str = "avail_physical_qubits",
        description: str = "Total number of available physical qubits",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of physical qubits available in both modules.

        :param experiment: experiment to calculate the available physical qubits for.
        """
        return 2 * experiment.system_architecture.qpu.num_qubits


class AvailLogicalQubits(Resource):
    """Total number of logical qubits available in the both modules of the FTQC."""

    def __init__(
        self,
        name: str = "Total number of available logical qubits",
        short_name: str = "avail_logical_qubits",
        description: str = "Total number of available logical qubits",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of available logical qubits.

        :param experiment: experiment to calculate the available logical qubits for.
        """
        return experiment.available_logical_qubits


class UnallocLogicalQubits(Resource):
    """Minimum number of left-over unallocated logical qubits in each of the modules considering all operations."""

    def __init__(
        self,
        name: str = "Minimum number of unallocated logical qubits",
        short_name: str = "unalloc_logical_qubits",
        description: str = "Minimum number of unallocated logical qubits",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of unallocated qubits per fridge.

        :param experiment: experiment to calculate the unallocated logical qubits for.
        """
        return experiment.available_logical_qubits - (experiment.graph_items.delta or 0)


class DistStageDecodingCores(Resource):
    """Maximum number of concurrent decoding cores required at the distill-inject-consume stage."""

    def __init__(
        self,
        decoder_tock: DecoderTock,
        name: str = "Distill Decoding Cores",
        short_name: str = "distill_concurrentcores_decoding",
        description: str = "Number of concurrent decoding cores at distillation-consumption stage",
    ) -> None:
        """
        :param decoder_tock: decoder tock (secs).
        """
        super().__init__(name, short_name, description)
        self.decoder_tock = decoder_tock

    def get_value(self, experiment: Experiment) -> int:
        """Return number of decoding cores used at distillation stage.

        We assume that the TN decoder of [Chubb 2021, arXiv:2101.04125] is to be used. The scaling relations were
        derived from the same reference.

        :param experiment: experiment to calculate the decoder tock for.
        """
        q_tock = 8 * experiment.distance * experiment.system_architecture.intra_qcycle_sec
        return math.ceil(self.decoder_tock.get_value(experiment) / q_tock)


class DistillStageDecodingMemory(Resource):
    """Memory required for decoding at distillation-consumption stage as the (sub)graph is consumed."""

    def __init__(
        self,
        distill_concurrent_cores: DistStageDecodingCores,
        name: str = "Distillation decoding memory",
        short_name: str = "distill_decoding_maxmem_MB",
        description: str = "Max decoding memory required at distillation stage (MBytes)",
        unit: str = "MBytes",
    ) -> None:
        """
        :param distill_concurrent_cores: number of CPU cores required for distillation.
        """
        super().__init__(name, short_name, description, unit)
        self.distill_concurrent_cores = distill_concurrent_cores

    def get_value(self, experiment: Experiment) -> float:
        """Maximum random access memory used in MBytes at any given time for distillation stage.

        :param experiment: experiment to calculate the decoder memory for.
        """
        bond_dim = experiment.bond_dim
        num = 2 * find_physq_perpatch(experiment.distance) - 1
        return (
            self.distill_concurrent_cores.get_value(experiment)
            * (num + math.sqrt(num) * math.pow(bond_dim, 2))
            * 8
            * 1e-6
        )


class ChipArea(Resource):
    """Total area required for the QPU."""

    def __init__(
        self,
        name: str = "Chip area",
        short_name: str = "chip_area_sqm",
        description: str = "Area of quantum chip (all modules)",
        unit="m^2",
    ) -> None:
        super().__init__(name, short_name, description, unit)

    def get_value(self, experiment: Experiment) -> float:
        """Return total QPU chip area for all modules.

        :param experiment: experiment to calculate the chip area for.
        """
        return experiment.system_architecture.qpu.processor_area_sqmm * 1e-6 * 2


class NumberCouplers(Resource):
    """The total number of required couplers."""

    def __init__(
        self,
        name: str = "Number of couplers",
        short_name: str = "number_of_couplers",
        description: str = "Number of couplers",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return number of required couplers.

        :param experiment: experiment to calculate the number of couplers for.
        """
        return experiment.system_architecture.qpu.num_couplers * 2


class DecodingDistillationPower(Resource):
    """Decoding power used during distillation-consumption stage.

    Based on a 100W reference CPU units.
    """

    def __init__(
        self,
        dist_cores: DistStageDecodingCores,
        name: str = "Distill Stage Decoding Power",
        short_name: str = "distill_decoding_power_kW",
        description: str = "Decoding power at distillation stage (kW)",
        unit="kW",
    ) -> None:
        """
        :param dist_cores: number of CPU cores used for distillation.
        """
        super().__init__(name, short_name, description, unit)
        self.dist_cores = dist_cores

    def get_value(self, experiment: Experiment) -> float:
        """Return decoding power used during distillation.

        :param experiment: experiment to calculate the decoding power for.
        """
        return self.dist_cores.get_value(experiment) * 100 * 0.001


class PowerDissipation(Resource):
    """Power dissipation at 4K stage."""

    def __init__(
        self,
        thermal_loads_index: int = 0,
        name: str = "power_dissipation_4K_kW",
        short_name: str = "power_dissip_4K_kW",
        unit: str = "kW",
        description: str = "Total power dissipation during the 4K stage (kW)",
    ) -> None:
        """
        :param thermal_loads_index: integer to specify which thermal load to use. 4K: tl_index=0, or mxc: tl_index=1
        """
        super().__init__(name, short_name, description, unit)
        self.tl_index = thermal_loads_index

    def get_value(self, experiment: Experiment) -> float:
        """Return power dissipation at the 4K (tl_index=0) or mxc (tl_index=1) stages.

        :param experiment: experiment to calculate the power dissipation for.
        """
        therm_loads = experiment.system_architecture.calc_thermal_loads()
        return (therm_loads[self.tl_index].power_dissipation_watt or 0) * 0.001


class TotalIntraQOpsTime(Resource):
    """Total time for the intra-module surface code ops required for graph consumption (excludes T-state transfers).

    Here, we assume a staggered architecture where there are enough multiple reference digital units to processes
    overlapping decoding tasks for a single widget. Therefore per-widget wall-time is only delayed by an overall
    decoding delay.
    """

    def __init__(
        self,
        name: str = "Total intra-module quantum ops time",
        short_name: str = "tot_intra_q_ops_sec",
        description: str = "Total intra-module quantum ops time (sec)",
        unit="sec",
    ) -> None:
        super().__init__(name, short_name, description, unit)

    def get_value(self, experiment: Experiment) -> float:
        """Return the total time taken by intra-module quantum ops.

        :param experiment: experiment to calculate the total intra Q ops time for.
        """
        return experiment.consump_sched_times.intra_consump_ops_time_sec


class TotalTStateTime(Resource):
    """Total intra-module time to distill and inject T-states for surface code ops and parity measurements."""

    def __init__(
        self,
        name: str = "Total intra-module time for T-state transfers",
        short_name: str = "tot_t_state_time_sec",
        description: str = "Total intra-module T-state transfer time (sec)",
        unit: str = "sec",
    ) -> None:
        super().__init__(name, short_name, description, unit)

    def get_value(self, experiment: Experiment) -> float:
        """Return the module execution time.

        :param experiment: experiment to calculate the total T state time for.
        """
        return experiment.consump_sched_times.intra_t_injection_time_sec


class TotalHandoverTime(Resource):
    """Total subgraph handover time."""

    def __init__(
        self,
        name: str = "Total handover time",
        short_name: str = "total_handover_time_sec",
        description: str = "Total graph handover time (sec)",
        unit="sec",
    ) -> None:
        super().__init__(name, short_name, description, unit)

    def get_value(self, experiment: Experiment) -> float:
        """Return the total handover time.

        :param experiment: experiment to calculate the total handover time for.
        """
        return experiment.consump_sched_times.inter_handover_ops_time_sec


class OverallDecodingDelay(Resource):
    """Overall decoding delay."""

    def __init__(
        self,
        decoder_tock: DecoderTock,
        total_intra_qops_time: TotalIntraQOpsTime,
        total_t_state_time: TotalTStateTime,
        name: str = "Overall decoding delay",
        short_name: str = "overall_decoding_delay_sec",
        description: str = "Overall decoding delay (fully classical, sec)",
        unit: str = "sec",
    ) -> None:
        """
        :param decoder_tock: decoder tock.
        :param tot_intra_qops_time: Total time for algorithmic surface code ops and parity meas for graph consumption.
        :param total_t_state_time: Total T state distillation and injection time.
        """
        super().__init__(name, short_name, description, unit)
        self.decoder_tock = decoder_tock
        self.total_intra_qops_time = total_intra_qops_time
        self.total_t_state_time = total_t_state_time

    def get_value(
        self,
        experiment: Experiment,
    ) -> float:
        """Return the overall decoding delay.

        :param experiment: experiment to calculate the overall decoding delay for.
        """
        dec_tock = self.decoder_tock.get_value(experiment)
        q_tock = 8 * experiment.distance * experiment.system_architecture.intra_qcycle_sec
        magicstate_tock = (
            8 * (experiment.distance + experiment.widget.cycles) * experiment.system_architecture.intra_qcycle_sec
        )
        tot_distill_delay_sec = 0.0
        reps_consump = math.ceil(self.total_intra_qops_time.get_value(experiment) / q_tock)
        reps_magicstate = math.ceil(self.total_t_state_time.get_value(experiment) / magicstate_tock)
        if dec_tock > q_tock:  # If dec_tock is longer, we need to add the delay for each consumption cycle
            tot_distill_delay_sec += reps_consump * (dec_tock - q_tock)
        if dec_tock > magicstate_tock:  # If dec_tock is longer, we need to add the delay for each distill-inject cycle
            tot_distill_delay_sec += reps_magicstate * (dec_tock - magicstate_tock)
        return tot_distill_delay_sec


class TotalFTTime(Resource):
    """Total time considering all non-simultaneous ops of the FTQC."""

    def __init__(
        self,
        quant_tock: QuantumIntraTock,
        total_intra_qops_time: TotalIntraQOpsTime,
        total_handover_time: TotalHandoverTime,
        dec_delay: OverallDecodingDelay,
        total_t_state_time: TotalTStateTime,
        name: str = "Total FT Time",
        short_name: str = "total_ft_time_sec",
        description: str = "Total FT-hardware wall-time (upper bound, sec)",
        unit="sec",
    ) -> None:
        """
        :param quant_tock: quantum tock for algorithmic surface code ops.
        :param tot_intra_qops_time: total time for algorithmic surface code ops and parity meas for graph consumption.
        :param total_handover_time: total subgraph handover time.
        :param dec_delay: total time delay for decoding.
        :param total_t_state_time: total T state distillation and injection time.
        """
        super().__init__(name, short_name, description, unit)
        self.total_intra_qops_time = total_intra_qops_time
        self.dec_delay = dec_delay
        self.total_t_state_time = total_t_state_time
        self.total_handover_time = total_handover_time
        self.quantum_tock = quant_tock

    def get_value(
        self,
        experiment: Experiment,
    ) -> float:
        """Return the total FT time.

        :param experiment: experiment to calculate the total fault tolerant time for.
        """
        prep_sched = experiment.graph_items.prep_sched if experiment.graph_items.prep_sched else [[]]
        first_widget_prep_time = len(prep_sched[0]) * self.quantum_tock.get_value(experiment)
        return (
            first_widget_prep_time
            + self.total_intra_qops_time.get_value(experiment)
            + self.total_t_state_time.get_value(experiment)
            + self.total_handover_time.get_value(experiment)
            + self.dec_delay.get_value(experiment)
        )


class FullEvolutionTime(Resource):
    """Total evolution time."""

    def __init__(
        self,
        ft_time: TotalFTTime,
        name: str = "Total evolution time",
        short_name: str = "full_evolution_time_sec",
        description: str = "Full evolution time (sec)",
        unit: str = "sec",
    ) -> None:
        """
        :param ft_time: total FT-hardware time
        """
        super().__init__(name, short_name, description, unit)
        self.ft_time = ft_time

    def get_value(self, experiment: Experiment) -> float:
        """Return the total evolution time.

        :param experiment: experiment to calculate the total evolution time for.
        """
        return experiment.num_steps * self.ft_time.get_value(experiment)


class TotalFTEnergy(Resource):
    """Total consumed power."""

    def __init__(
        self,
        distill_concurrent_cores: DistStageDecodingCores,
        avail_phys_qubits: AvailPhysicalQubits,
        req_phys_qubits: ReqPhysicalQubits,
        total_ft_time: TotalFTTime,
        decoder_tock: DecoderTock,
        power_4k: PowerDissipation,
        power_mxc: PowerDissipation,
        name: str = "Total FT energy",
        short_name: str = "total_ft_energy_kWh",
        description: str = "Total FT-hardware energy consumption (kWh)",
        unit: str = "kWh",
    ) -> None:
        """
        :param distill_concurrent_cores: no. of concurrent decoding cores required at the dist-consump stage.
        :param avail_phys_qubits: no. of available physical qubits
        :param req_phys_qubits: no. of required physical qubits
        :param total_ft_time: total fault tolerant computation time
        :param decoder_tock: decoder tock (sec)
        :param power_4k: power dissipation at the 4K stage
        :param power_mxc: power dissipation at the MXC stage
        """
        super().__init__(name, short_name, description, unit)
        self.distill_concurrent_cores = distill_concurrent_cores
        self.avail_phys_qubits = avail_phys_qubits
        self.req_phys_qubits = req_phys_qubits
        self.total_ft_time = total_ft_time
        self.decoder_tock = decoder_tock
        self.power_4k = power_4k
        self.power_mxc = power_mxc

    def get_value(
        self,
        experiment: Experiment,
    ) -> float:
        """Return total fault-tolerant hardware energy consumption.

        :param experiment: experiment to calculate the total energy consumption for.
        """
        cooling_efficiency_4kelvin = 500  # W/W cooling for a 4K cryo-cooler
        cooling_efficiency_mxc = 1e9  # W/W cooling for a dilution refrigerator at 0.02K
        repetitions = ((experiment.widget.cycles / experiment.distance) + 1) * experiment.graph_items.t_count()
        ref_decode_watt = 100
        total_ft_time = self.total_ft_time.get_value(experiment)
        decode_distill_energy = (
            self.distill_concurrent_cores.get_value(experiment)
            * ref_decode_watt
            * repetitions
            * self.decoder_tock.get_value(experiment)
        )
        cooling_4k_energy = (
            self.power_4k.get_value(experiment) * total_ft_time * cooling_efficiency_4kelvin * 1000
        )  # already in kW, convert to W
        cooling_mxc_energy = (
            self.power_mxc.get_value(experiment) * total_ft_time * cooling_efficiency_mxc * 1000
        )  # already in kW, convert to W

        # Note: Powers are not essentially the wall power, therefore, we need to consider efficiency
        return (decode_distill_energy + cooling_4k_energy + cooling_mxc_energy) / 1000 / 3600


class ResourceEstimator:
    """Class to coordinate the estimation of resources passed during initialization.

    This class also offers methods to output the results as a dictionary, CSV file, and/or to console via print
    statements.
    """

    def __init__(self, circuit_name: str, resources: Sequence[Resource], experiment: Experiment) -> None:
        """

        :param circuit_name: name of the circuit the estimate is being performed for.
        :param resources: list of resource instances to calculate estimates for.
        :param experiment: experiment class which contains the parameters and specifications for this estimation.
        """
        self.circuit_name = circuit_name
        self.resources = list(resources)
        self.experiment = experiment

    @property
    def formats(self) -> Dict[str, ResourceFormat]:
        """Return the formats for the resources, for use when printing results."""
        return {resource.short_name: resource.fmt for resource in self.resources}

    def to_dict(self, key: Literal["name", "short", "description"]) -> Dict[str, Optional[ResourceValue]]:
        """Return the experiment attributes and estimated resources as a dictionary.

        :param key: modifier to set how the result names are returned.
        """
        experiment_attrs = {
            "graph_state_opt": self.experiment.graph_state_opt,
            "est_method": self.experiment.est_method,
            "sims_time_sec": self.experiment.sims_time_sec,
            "version": self.experiment.version,
            "decomp_method": self.experiment.decomp_method,
            "target_p_algo": self.experiment.target_p_algo,
            "dist_widget_name": self.experiment.widget.name,
            "num_intermodule_pipes": self.experiment.system_architecture.num_intermodule_pipes,
            "num_timesteps": self.experiment.num_steps,
        }

        if key == "short":
            for resource in self.resources:
                logger.debug(
                    f"The following resource was added: {resource.short_name}: {resource.get_value(self.experiment)}"
                )
            return {
                **experiment_attrs,
                **self.experiment.circuit_info,
                **{resource.short_name: resource.get_value(self.experiment) for resource in self.resources},
            }
        if key == "name":
            return {resource.name: resource.get_value(self.experiment) for resource in self.resources}
        if key == "description":
            return {resource.description: resource.get_value(self.experiment) for resource in self.resources}

    def to_console(self, key="description"):
        """Print results to console."""
        print(
            "RRE: Estimated fault-tolerant resources required for the two fridge bilinear-q-bus architecture with "
            "gate-synth at the measurement points:\n"
        )
        for desc, value in self.to_dict(key=key).items():  # type: ignore
            print(f"\t{desc}: {value}")

    def to_csv(self, file_name: str, mode: str = "w") -> None:
        """Save resources to CSV.

        :param file_name: path and file name to save the results to.
        :param mode: mode to write csv. a - append, w - write
        """
        df = pd.DataFrame(self.to_dict(key="short"), index=[self.circuit_name])
        df.index.name = "circuit"
        for col, dtype in self.formats.items():
            df[col] = df[col].astype(dtype)

        if mode == "a":
            df.to_csv(file_name, mode=mode, header=False)
        elif mode == "w":
            df.to_csv(file_name, mode=mode)
        else:
            raise ValueError("CSV mode must be either 'a' for append, or 'w' for write.")


class ResourceCollection:
    """A class to build a list of resources for estimation purposes."""

    @property
    @abstractmethod
    def name(self):
        """Return the name of this ResourceCollection."""

    @abstractmethod
    def build(self) -> Sequence[Resource]:
        """Create a list of resources."""


class DefaultResourceCollection(ResourceCollection):
    """The default ResourceCollection class."""

    @property
    def name(self):
        """Return the name of this ResourceCollection."""
        return "Default"

    def build(self):
        """Create the default list of resources."""
        resources = []
        resources.append(DistillWidgetQubits())
        resources.append(Distance())
        resources.append(RequiredLogicalQubits())
        resources.append(NumTFactories())
        resources.append(ReqQubitsTFactories())
        resources.append(req_phys_q := ReqPhysicalQubits())
        resources.append(TCount())
        resources.append(avail_phys_q := AvailPhysicalQubits())
        resources.append(MaxIONodes())
        resources.append(decoder_tock := DecoderTock())
        resources.append(quant_tock := QuantumIntraTock())
        resources.append(TStateTock())
        resources.append(dist_cores := DistStageDecodingCores(decoder_tock=decoder_tock))
        resources.append(tot_q_ops_time := TotalIntraQOpsTime())
        resources.append(tot_t_state_time := TotalTStateTime())
        resources.append(tot_handover_time := TotalHandoverTime())
        resources.append(
            decode_delay := OverallDecodingDelay(
                decoder_tock=decoder_tock,
                total_intra_qops_time=tot_q_ops_time,
                total_t_state_time=tot_t_state_time,
            )
        )
        resources.append(power_4k := PowerDissipation(thermal_loads_index=0))
        resources.append(power_mxc := PowerDissipation(thermal_loads_index=1))
        resources.append(AvailLogicalQubits())
        resources.append(UnallocLogicalQubits())
        resources.append(InputLogicalQubits())
        resources.append(DiamondNormEps())
        resources.append(TDepth())
        resources.append(InitRZCount())
        resources.append(InitTCount())
        resources.append(InitCliffordCount())
        resources.append(GraphN())
        resources.append(ConsumpScheduleSteps())
        resources.append(PrepScheduleSteps())
        resources.append(DistillStageDecodingMemory(distill_concurrent_cores=dist_cores))
        resources.append(ChipArea())
        resources.append(NumberCouplers())
        resources.append(DecodingDistillationPower(dist_cores=dist_cores))
        resources.append(
            ft_time := TotalFTTime(
                total_intra_qops_time=tot_q_ops_time,
                dec_delay=decode_delay,
                total_t_state_time=tot_t_state_time,
                quant_tock=quant_tock,
                total_handover_time=tot_handover_time,
            )
        )
        resources.append(FullEvolutionTime(ft_time=ft_time))
        resources.append(
            TotalFTEnergy(
                distill_concurrent_cores=dist_cores,
                avail_phys_qubits=avail_phys_q,
                req_phys_qubits=req_phys_q,
                total_ft_time=ft_time,
                decoder_tock=decoder_tock,
                power_4k=power_4k,
                power_mxc=power_mxc,
            )
        )
        return resources
