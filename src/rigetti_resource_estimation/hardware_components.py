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
**Module** ``hardware_components``

A collection of classes to express or calculate hardware configurations given a fault-tolerant system architecture.
"""

from dataclasses import dataclass, field
from enum import Enum

from typing import List, Optional, NamedTuple
import math
import copy
import logging

import pandas as pd
import numpy as np

from rigetti_resource_estimation.graph_utils import CompilerSchedGraphItems

logger = logging.getLogger(__name__)


ConsumpSchedTimes = NamedTuple(
    "ConsumpSchedTimes",
    [
        ("intra_consump_ops_time_sec", float),
        ("intra_distill_delay_sec", float),
        ("intra_prep_delay_sec", float),
        ("inter_handover_ops_time_sec", float),
    ],
)

IntraComponentCounts = NamedTuple(
    "IntraComponentCounts",
    [
        # Logical edge size for all modules:
        ("edge_logical", int),
        # Logical length of the quantum bus in the dir perpendicular to the boundary of the quantum and T-transfer bus:
        ("l_qbus", int),
        # Number of distillation columns in each module:
        ("c_factories", int),
        # Number of T-distillation factories per module:
        ("num_t_factories", int),
        # Logical length og the T-transfer bus:
        ("len_transfer_bus", int),
        # Number of required logical qubits per module for all intra-module components:
        ("num_logical_qubits", int),
        # Unallocated logical qubits per module after assigning the nodes of all intra-module components:
        ("unalloc_logical_qubits", int),
    ],
)


class LineType(Enum):
    """A class for tracking line types for subsequent resource estimate tabulation."""

    MICROWAVE_XY = 0
    QUBIT_FLUX = 1
    COUPLER_FLUX = 2
    READOUT_IN = 3
    READOUT_OUT = 4


@dataclass
class CountNodesOverModules:
    """A class to keep track of maximum no. of and crossing for T, Rz, and I/O nodes given FTQC's ladder arch."""

    nodes_in_modules: list
    delta: int
    num_modules_per_leg: int

    @classmethod
    def from_fridges_per_leg(cls, delta: int, num_modules_per_leg: int):
        """Add nodes to `nodes_in_modules` based on `delta` and `num_modules_per_leg` data."""
        nodes_in_modules = []
        for ii in range(num_modules_per_leg):
            nodes = list(
                range(
                    ii * math.ceil(delta / num_modules_per_leg),
                    min(delta, (ii + 1) * math.ceil(delta / num_modules_per_leg)),
                )
            )
            logger.info(f"\nFor module {ii}, RRE will add {nodes} to `nodes_in_modules`.\n")
            nodes_in_modules.append(nodes)
        return cls(nodes_in_modules, delta, num_modules_per_leg)

    def n_max_nodes(self, meas_sublist: list) -> int:
        """
        Calculate maximum no. of requested nodes of the same measurement type in `meas_sublist` over all modules given.

        We currently consider a naive and suboptimal T-routing and the placement of T and Rz-measure intermediate nodes
        in delta positions across `num_modules_per_leg` modules. We will locally minimize the current `n_max_nodes`
        nodes in `meas_sublist` (ignoring optimizations for other sub-step nodes) by placing intermediate nodes of with
        the same measurement basis in as many distinct modules as possible.
        """
        n_max_nodes = math.ceil(len(meas_sublist) / self.num_modules_per_leg)
        last_num_capacity = math.floor(len(meas_sublist) / self.num_modules_per_leg)
        if self.num_modules_per_leg > 1 and last_num_capacity > len(self.nodes_in_modules[-1]):
            n_max_nodes = n_max_nodes + math.ceil(
                (last_num_capacity - len(self.nodes_in_modules[-1])) / (self.num_modules_per_leg - 1)
            )
        if n_max_nodes == 0 and meas_sublist:
            raise AssertionError(
                "For the following measurement_sublist data:\n"
                f"\tmeas_sublist:{meas_sublist},\n\tdelta={self.delta},\n\tnum_modules_per_leg={self.num_modules_per_leg},\n"
                f"\tRRE detected inconsistent results as nodes_in_modules={self.nodes_in_modules} and n_max_nodes=0.\n"
            )
        else:
            logger.debug(
                "For the following measurement_sublist data:\n"
                f"\tmeas_sublist:{meas_sublist},\n\tdelta={self.delta},\n\tnum_modules_per_leg={self.num_modules_per_leg},\n"
                f"\tRRE detected nodes_in_modules={self.nodes_in_modules} and n_max_nodes={n_max_nodes}.\n"
            )
        return n_max_nodes

    def num_crossmodule_io(self, output_nodes: list, input_nodes: list) -> int:
        """
        Calculate min no. of module crossing required to teleport all nodes from `output_nodes` to `input_nodes`.

        We calculate the minimum no. of crossing required for the simplest case scenario of `output_nodes` and
        `input_nodes` are assigned to delta memory positions according to the ordering of original compilation indices.
        """
        n_crossfrdige_io = len(output_nodes)  # this amount of crossings is always needed to go from one leg to another
        for index, _ in enumerate(output_nodes):
            n_crossfrdige_io += min(
                math.floor(
                    np.abs(output_nodes[index] - input_nodes[index]) / math.ceil(self.delta / self.num_modules_per_leg)
                ),
                self.num_modules_per_leg - 1,  # worst-case scenario, we need (n_fridges_per_leg - 1) same-leg crossing
            )
        logger.debug(
            f"\nFor the output_nodes={output_nodes}, input_nodes={input_nodes}, delta={self.delta}, and "
            f"num_modules_per_leg={self.num_modules_per_leg}, RRE detected n_crossfrdige_io={n_crossfrdige_io}.\n"
        )
        return n_crossfrdige_io


@dataclass
class SeqPrepOps:
    """
    A class to keep track of no. of sequential operations over a single preparation schedule step.

    :param prep_subschedules: a single step from the preparation schedule (from substrate scheduler) required for the
        execution of stabilizer measurements to initialize the graph.
    :param delta: maximum number of logical qubits (max memory), required at any moment.
    :param fridge_crosses_list: a list storing how many fridge crosses happen for every prep subschedule.
    :param num_modules_per_leg: no. of modules per each leg of the ladder macro-architecture.
    :param num_pipes_per_intermodule_connection: no. of width-d interconnect lines connecting a pair of modules.
    """

    prep_subschedules: List[list]
    delta: int
    fridge_crosses_list: list = field(default_factory=list)
    num_modules_per_leg: int = 1
    num_pipes_per_intermodule_connection: int = 1

    def __post_init__(self):
        """Post-process some of the initial attributes."""
        for prep_subschedule in self.prep_subschedules:
            fridge_crosses = 0
            for prep_tuples in prep_subschedule:
                stab_ind = prep_tuples[0]
                min_ind = prep_tuples[1][0]
                max_ind = prep_tuples[1][1]
                max_distance = max(stab_ind, min_ind, max_ind) - min(stab_ind, min_ind, max_ind)
                if max_distance >= self.delta:
                    fridge_crosses += 1
            # worst-case scenario, we need (n_fridges_per_leg - 1) same-leg crossing
            self.fridge_crosses_list.append(min(fridge_crosses, self.num_modules_per_leg - 1))

    def seq_fridge_crosses(self) -> int:
        """Calculate no. of sequential cross-fridge ops required to execute the prep step `prep_subschedules`."""
        seq_fridge_crosses = 0
        if self.num_modules_per_leg > 1:
            for ii, _ in enumerate(self.prep_subschedules):
                seq_fridge_crosses += math.ceil(
                    self.fridge_crosses_list[ii] / self.num_pipes_per_intermodule_connection
                )
        return seq_fridge_crosses

    def seq_intra_comms(self) -> int:
        """Calculate no. of sequential intra-fridge ops required to execute the prep step `prep_subschedules`."""
        seq_intra_comms = 0
        if self.num_modules_per_leg > 1:
            for ii, prep_subschedule in enumerate(self.prep_subschedules):
                if len(prep_subschedule) > self.fridge_crosses_list[ii]:
                    seq_intra_comms += 1
        else:
            seq_intra_comms = len(self.prep_subschedules)
        return seq_intra_comms


@dataclass
class Component:
    """A generic class for holding a component, which can be used for resource estimations subsequently.

    :param name: name of the component.
    """

    name: str

    @classmethod
    def from_dict(cls, kwargs):
        """Create a Component instance from dictionary of parameters."""
        return cls(**kwargs)


@dataclass
class DistWidget:
    """A class to keep track of distillation widgets.

    :param name: name of the distillation widget.
    :param cycles: number of cycles.
    :param p_out: output (logical) failure rate.
    :param qubits: number of qubits.
    :param cond_20to4: is 20 to 4 widget.
    """

    name: str
    cycles: float
    p_out: float
    qubits: int
    cond_20to4: bool
    width: int
    length: int


@dataclass
class WidgetTable:
    """A list of distillation widgets creating the lookup table.

    :param widgets: list of widgets.
    """

    widgets: List[DistWidget]

    @classmethod
    def from_dict(cls, dct):
        """Create a DistWidgetTable instance from dictionary of parameters."""
        widgets = [DistWidget(**widget) for widget in dct["widgets"]]
        return cls(widgets=widgets)


@dataclass
class CryostatStage:
    """Stage of a cryostat with components in a signal chain.

    :param temperature_kelvin: temperature in kelvin.
    :param power_dissipation_watt: power dissipation in watts.
    :param cooling_power_watt: cooling power in watts.
    :param component: component.
    """

    temperature_kelvin: float
    power_dissipation_watt: Optional[float] = None
    cooling_power_watt: Optional[float] = None
    component: Optional[Component] = None

    @classmethod
    def from_dict(cls, dct):
        """Create a CryostatStage instance from dictionary of parameters."""
        spec = copy.deepcopy(dct)
        component_def = spec.pop("component")
        if component_def is not None:
            component_cls = globals()[component_def.pop("type")]
            component = component_cls(**component_def)
        else:
            component = None
        return cls(**spec, component=component)


@dataclass
class Fridge:
    """Parent class for fridge modules in a quantum computing device.

    Holds the definition of fridges such as Bluefors, Bluefors LD, or Bluefors XLD.

    :param name: Name of the fridge.
    :param stages: A list of cryostat stages contained within the fridge.
    """

    name: str
    stages: List[CryostatStage]

    @classmethod
    def from_dict(cls, dct):
        """Create a Fridge instance from dictionary of parameters."""
        name = dct["name"]
        stages = [CryostatStage(**stage) for stage in dct["stages"]]
        return cls(name=name, stages=stages)


@dataclass
class Cable(Component):
    """A component representing a microwave cable.

    :param length_m: Electrical length of the cable in meter.
    """

    length_m: float


@dataclass
class Attenuator(Component):
    """A component representing a microwave attenuators.

    :param att_decibel: Attenuation in dB.
    """

    att_decibel: float


@dataclass
class Circulator(Component):
    """Class for Circulators' attributes.

    :param isolation_decibel: Reverse isolation of the circulator.
    :param loss_decibel: Forward attenuation of the circulator.
    """

    isolation_decibel: float
    loss_decibel: float


@dataclass
class Amplifier(Component):
    """A component for representing an amplifier.

    :param gain_decibel: The gain of the amplifier in dB.
    :param noise_kelvin: Noise temperature of the amplifier in Kelvin.
    """

    gain_decibel: float
    noise_kelvin: float


@dataclass
class SignalChain:
    """Definition of a signal chain to carry signals to/from the qubit to room temp.

    :param name: Label for the signal chain for tracking purposes.
    :param component_current: The current supplied to the line, as measured at the powered device
        be that an amplifier or the qubit. For lines that supply current (flux bias, TWPA drive),
        this is the current at the qubit level.
    :param line_type: The functional type of the signal line.
    :param components: The list of components that comprise the chain, in order of signal propagation.
    """

    name: str
    component_current: float
    line_type: LineType
    components: List

    @classmethod
    def from_dict(cls, dct):
        """Create a SignalChain instance from dictionary of parameters."""
        spec = copy.deepcopy(dct)
        component_list = spec.pop("components")
        components = []
        for component_def in component_list:
            component_cls = globals()[component_def.pop("type")]
            components.append(component_cls(**component_def))
        return cls(**spec, components=components)


@dataclass
class QuantumProcessor:
    """A class for counting output parameters relating to the quantum integrated circuit.

    We assume a layout of logical qubit as selected by the T-factory from Widget Lookup Table in `params.yaml` and
    that the Quantum Processor has square geometry.

    :param num_qubits: number of physical qubits.
    :param num_couplers: number of couplers.
    :param num_junctions: number of junctions.
    :param num_readout_lines: number of readout lines.
    :param num_xy_lines: number of XY lines.
    :param num_qubit_flux_lines: number of qubit flux lines.
    :param num_coupler_flux_lines: number of coupler flux lines.
    :param qubit_pitch_mm: qubit pitch measured in mm.
    :param processor_area_sqmm: QPU area measured in mm^2.
    :param qubits_per_readout_line: number of physical qubits per readout lines.
    :param max_qubits_per_logical_qubit: maximum number of physical qubits supported for each logical one.
    """

    num_qubits: int
    num_couplers: int
    num_junctions: int
    num_readout_lines: int
    num_xy_lines: int
    num_qubit_flux_lines: int
    num_coupler_flux_lines: int
    qubit_pitch_mm: float
    processor_area_sqmm: float
    qubits_per_readout_line: int
    max_qubits_per_logical_qubit: int

    @classmethod
    def from_qubit_number(
        cls,
        widget: DistWidget,
        num_qubits: int = 1000000,
        pitch_mm: float = 2.0,
        qubits_per_readout_line: int = 10,
    ):
        """Create a QuantumProcessor from fewer inputs, subject to some assumptions.

        :param widget: the distillation widget (T-factory) in the form of an DistWidget object.
        :param num_qubits: number of physical qubits.
        :param pitch_mm: qubit pitch measured in mm.
        :param qubits_per_readout_line: number of physical qubits per readout line.
        """
        num_couplers = 2 * num_qubits
        num_junctions = num_couplers * 2 + num_qubits * 2
        if num_qubits % qubits_per_readout_line == 0:
            num_readout_lines = int(num_qubits / qubits_per_readout_line)  # Do we need to check for partial lines?
        else:
            raise ValueError("Not an even number of qubits per readout line")
        num_xy_lines = int(num_qubits)
        num_qubit_flux_lines = int(num_qubits)
        num_coupler_flux_lines = int(num_couplers)
        processor_area_sqmm = pitch_mm**2 * num_qubits
        max_qubits_per_logical_qubit = math.floor((num_qubits - widget.qubits) / 2)

        if max_qubits_per_logical_qubit < 18:
            raise ValueError(
                f"Maximum possible qubits per logical ones was {max_qubits_per_logical_qubit}. "
                "QPU is not big enough to support a single logical qubit considering at least 18 qubits are required "
                "for smallest possible d."
            )

        return cls(
            num_qubits=num_qubits,
            num_couplers=num_couplers,
            num_junctions=num_junctions,
            num_readout_lines=num_readout_lines,
            num_xy_lines=num_xy_lines,
            num_qubit_flux_lines=num_qubit_flux_lines,
            num_coupler_flux_lines=num_coupler_flux_lines,
            qubit_pitch_mm=pitch_mm,
            processor_area_sqmm=processor_area_sqmm,
            qubits_per_readout_line=qubits_per_readout_line,
            max_qubits_per_logical_qubit=max_qubits_per_logical_qubit,
        )

    @classmethod
    def from_dict(cls, widget, dct):
        """Create a QuantumProcessor instance from dictionary of parameters.

        :param widget: the distillation widget (T-factory) in the form of an DistWidget object
        :param dct: a dictionary of parameters.
        """
        return cls.from_qubit_number(widget, **dct)


@dataclass
class FTQC:
    """A complete utility-scale fault-tolerant quantum computer.

    Combines the quantum processor definition with the wiring requirements to create a build out of the things needed
    to build out the system (currently, only for the stuff in the fridge). We assume a bilinear-bus two fridge
    architecture for the complete FTQC by default.

    :param name: name of the FTQC system.
    :param cryostat: th cryostat module in the form of a Fridge object.
    :param qpu: a QuantumProcessor object.
    :param xy_line: the XY line in the form of a SignalChain.
    :param qubit_flux_line: the qubit flux line in the form of a SignalChain.
    :param coupler_flux_line: the coupler flux line in the form of a SignalChain.
    :param readout_out_line: the readout's out line in the form of a SignalChain.
    :param readout_in_line: the readout's in line in the form of a SignalChain.
    :param inter_handover_timescale_sec: characteristic timescale to teleport states for subgraph handover/stitching.
    :param intra_qcycle_sec: characteristic timescale for all intra-module quantum operations.
    :param num_pipes_per_intermodule_connection: number of pipes connecting the two fridges.
    """

    name: str
    cryostat: Fridge
    qpu: QuantumProcessor
    xy_line: SignalChain
    qubit_flux_line: SignalChain
    coupler_flux_line: SignalChain
    readout_out_line: SignalChain
    readout_in_line: SignalChain
    inter_handover_timescale_sec: float
    intra_qcycle_sec: float
    num_pipes_per_intermodule_connection: int

    def calc_readout_resources(self):
        """Calculate output parameters related to readout."""
        return self.qpu.num_readout_lines * len(self.readout_out_line.components)

    def calc_thermal_loads(self, num_modules_per_leg: int = 1):
        """Calculate total thermal loads on cryostat modules from device parameters."""
        for stage in self.cryostat.stages:
            stage.power_dissipation_watt = 0
            for line, num in [
                (self.xy_line, self.qpu.num_xy_lines),
                (self.readout_out_line, self.qpu.num_readout_lines),
                (self.readout_in_line, self.qpu.num_readout_lines),
                (self.qubit_flux_line, self.qpu.num_qubit_flux_lines),
                (self.coupler_flux_line, self.qpu.num_coupler_flux_lines),
            ]:
                stage_list = []
                for comp in line.components:
                    if isinstance(comp, CryostatStage) is True:
                        stage_list.append(comp)
                df = pd.DataFrame(stage_list)
                stage.power_dissipation_watt += (
                    df[df["temperature_kelvin"] == stage.temperature_kelvin]["power_dissipation_watt"].sum()
                    * 2  # factor 2 comes from the fact that we have a two legs in the ladder-style macro-architecture
                    * num_modules_per_leg
                    * num
                )

        return self.cryostat.stages

    def calc_num_t_factories(
        self,
        dist_widget: DistWidget,
        d: int,
        delta: int,
        qpu_qubits: Optional[int] = None,
    ) -> IntraComponentCounts:
        """Calculate the number of available T-factories per fridge given intra-module FT-architecture details and d.

        This method also validate if different components of the intra-module architecture fits in the fridge given
        total number of physical qubits. See the RRE manuscript arXiv:2406.06015 for details of the intra-module
        FT-architecture.

        :param dist_widget: a DistWidget object specifying the distillation widgets (T-factories).
        :param d: code distance.
        :param delta: required no. of logical nodes in the (sub)graph states.
        :param qpu_qubit: total number of physical qubits available in each QPU module.

        :return: a IntraComponentCount object containing the counts of the internal components allocated to each module.
        """
        if not qpu_qubits:
            qpu_qubits = self.qpu.num_qubits
        edge_logical = math.floor(np.sqrt(qpu_qubits / (2 * d * d)))
        if edge_logical < 6:  # Minimally, 3 columns are needed for the bus, 1 for T-buffer area, and 2 for T-factories
            logger.warning(f"Physical qubit no. of {qpu_qubits} is too small to fit practical components.")

        widget_logical_l = math.ceil(dist_widget.length / (np.sqrt(2) * d))
        widget_logical_w = math.ceil(dist_widget.width / (np.sqrt(2) * d))
        l_qbus = max(math.ceil(delta / (2 * math.floor((edge_logical - 2) / 4) + 1)), 3)
        if edge_logical - l_qbus - 1 < widget_logical_l + 1:
            logger.debug(
                "The quantum, T-transfer bus and T-factories cannot be allocated to a fridge of logical size "
                f"{edge_logical}x{edge_logical}. RRE will set zero T-factories available based on the following data:\n"
                f"qpu.num_qubits: {qpu_qubits}, dist_widget.name: {dist_widget.name} (logical size of "
                f"{widget_logical_l}x{widget_logical_w}), d: {d}\n"
            )

        c_factories = max(math.floor((edge_logical - l_qbus - 1) / (widget_logical_l + 1)), 0)
        len_transfer_bus = max(
            (edge_logical - l_qbus - c_factories * widget_logical_l) * edge_logical + c_factories * widget_logical_l, 0
        )
        num_t_factories = max(math.floor((edge_logical - 1) / widget_logical_w) * c_factories, 0)
        num_logical_qubits = 2 * delta + len_transfer_bus + num_t_factories * widget_logical_l * widget_logical_w
        unalloc_logical_qubits = (
            edge_logical**2 - 2 * delta - len_transfer_bus - num_t_factories * widget_logical_l * widget_logical_w
        )

        logger.debug(
            "The following IntraComponentCounts were allocated:\n"
            f"\tedge_logical={edge_logical}, l_qbus={l_qbus}, c_factories={c_factories}, "
            f"len_transfer_bus={len_transfer_bus}, num_t_factories={num_t_factories}, "
            f"total_avail_logical_qubits={edge_logical**2}, num_logical_qubits={num_logical_qubits}, and "
            f"unalloc_logical_qubits={unalloc_logical_qubits}\n"
        )

        return IntraComponentCounts(
            edge_logical=edge_logical,
            l_qbus=l_qbus,
            c_factories=c_factories,
            num_t_factories=num_t_factories,
            len_transfer_bus=len_transfer_bus,
            num_logical_qubits=num_logical_qubits,
            unalloc_logical_qubits=unalloc_logical_qubits,
        )

    def calc_consump_sched_times(
        self,
        graph_items: CompilerSchedGraphItems,
        dist_widget: DistWidget,
        num_pipes_per_intermodule_connection: int,
        distance: int,
        transpiled_widgets: dict,
    ) -> ConsumpSchedTimes:
        """Calculate the time quantities given the graph schedules and all intra and inter-modular communications.

        The FTQC architecture has different layers. The macro-architecture includes two interconnected leg of modules
        creating a ladder pattern, where every modules hosts algorithmic time-ordered subcircuit widgets and an equal
        number of T-factories. The two legs run interleaving subgraphs. When the modules in one leg grow the graph by
        preparing a new subgraph inside, the other, in contrast, consumes the prior subgraph, which is always specified
        to take longer than any graph preparation.

        Summary of the intra-module FT-architecture with T-state parallelization and explicit component allocation:
        Each module is considered a rectangular grid of logical qubits/patches as the building block of all components.
        The width and length are set to be as close to each other as possible, and every patch contains 2*d*d physical
        qubits. Other than unallocated, wasted space, all fridges have four main components. The first two components
        come from the bilinear quantum bus of ancilla and graph-state/data logical qubits. We lay these out in a comb or
        snake-like pattern to fill in the longer side of the fridge. Third is the linear T-transfer bus; the last
        components are T-distillation widgets/factories. On the remaining side of the module, the T-transfer-bus
        sandwiches as many T-factories as possible and is laid out in a comb-like pattern again so that two columns or
        more must touch the long side of the ancilla quantum bus. The T transfer bus actively stores T-states from
        factories and queues them during all sub-schedules. As detailed below, the T-transfer bus can feed many parallel
        T-states to the quantum bus during the subgraph consumptions. For full details, please also look at the RRE
        manuscript [arXiv:2406.06015](https://arxiv.org/abs/2406.06015).

        Users may switch to a single-module architecture (with T-factories inside) by appropriately configuring
        `inter_handover_timescale_sec`, `qcycle_char_timescale_ns`, and `processor.num_qubits` in `params.yaml`. One can
        also set `num_pipes_per_intermodule_connection` to speed-up subgraph handover to any desired number.

        Here, we iterate through all scheduling steps, calculate hardware times and validating architectural assumptions
        at the same time.

        :param graph_items: A CompilerSchedGraphItems object holding the attributes and scheds of the full graph state.
        :param dist_widget: A DistWidget object specifying the distillation widgets (T-factories).
        :param num_pipes_per_intermodule_connection: number of pipes connecting the two fridges in the architecture.
        :param distance: surface code distance.
        :param transpiled_widgets: RRE standard dict containing transpiled widgets info.

        :returns: A ConsumpSchedTimes object containing:
            `intra_consump_ops_time_sec`: total intra-module time for surface code ops relating to the consumption of
            the subgraphs in the unit of sec.
            `intra_t_injection_time_sec`: total intra-module time for transferring the T-states to be used for the
            consumption of the subgraph in the same fridge in the unit of sec.
            `inter_handover_ops_time_sec`: total inter-module time for the handover ops stitching the subgraph belonging
            to the current graph state into the subgraph in the second module.
        """
        delta = int(graph_items.delta or 0)
        consump_schedule = graph_items.consump_sched
        compiler_tag_table = graph_items.compiler_tag_table or {}
        prep_schedule = graph_items.prep_sched if graph_items.prep_sched else [[]]
        measurement_basis_list = graph_items.measurement_basis_list if graph_items.measurement_basis_list else [[]]
        output_nodes = graph_items.output_nodes if graph_items.output_nodes else [[]]
        input_nodes = graph_items.input_nodes if graph_items.input_nodes else [[]]
        t_length_unit = graph_items.t_length_unit
        transpiled_widgets_keys = list(transpiled_widgets["widgets"].keys())

        if consump_schedule is None:
            logger.info("The consumption schedule was empty!")
            return ConsumpSchedTimes(
                intra_distill_delay_sec=0.0,
                intra_prep_delay_sec=0.0,
                intra_consump_ops_time_sec=0.0,
                inter_handover_ops_time_sec=0.0,
            )

        num_modules_per_leg = self.allocate_logicals_to_modules(dist_widget=dist_widget, d=distance, delta=delta)
        max_nodes_over_modules = CountNodesOverModules.from_fridges_per_leg(
            delta=delta, num_modules_per_leg=num_modules_per_leg
        )
        intra_components_counts = self.calc_num_t_factories(
            dist_widget=dist_widget, d=distance, delta=math.ceil(delta / num_modules_per_leg)
        )
        # No. of rows in the bi-linear quantum bus in each module (counting solely rows of whether data or ancilla type
        # of size more than 1):
        n_row_qbus = max(math.floor((math.ceil(delta / num_modules_per_leg) + 1) / intra_components_counts.l_qbus), 1)
        n2 = min(
            (
                intra_components_counts.num_t_factories * 4
                if dist_widget.cond_20to4
                else intra_components_counts.num_t_factories
            ),
            n_row_qbus,
        )
        if intra_components_counts.num_t_factories == 0:
            logger.info("There were no T-factories to process the schedules!")
            return ConsumpSchedTimes(
                intra_distill_delay_sec=0.0,
                intra_prep_delay_sec=0.0,
                intra_consump_ops_time_sec=0.0,
                inter_handover_ops_time_sec=0.0,
            )

        intra_consump_ops_time_sec = 0.0
        intra_prep_delay_sec = 0.0
        intra_distill_delay_sec = 0.0
        inter_handover_ops_time_sec = 0.0

        for wid_index, widget_key in enumerate(transpiled_widgets_keys):
            logger.debug(
                f"Calculating time quantities for widget,index:{widget_key},{wid_index} and following schedules:\n"
                f"\tprep (sub)schedule: {prep_schedule[wid_index]}\n"
                f"\tmeasurement basis list: {measurement_basis_list[wid_index]}\n"
                f"\tconsumption (sub)schedule: {consump_schedule[wid_index]}\n"
            )

            t_angle_tags = [dict["angle_tag"] for op, dict in compiler_tag_table.items() if op in ["T", "T**-1"]]
            t_measures = [ind for ind, angle in enumerate(measurement_basis_list[wid_index]) if angle in t_angle_tags]

            rot_angle_tags = [dict["angle_tag"] for op, dict in compiler_tag_table.items() if op in ["ArbRz"]]
            rot_measures = [
                ind for ind, angle in enumerate(measurement_basis_list[wid_index]) if angle in rot_angle_tags
            ]

            cumulative_t_intra_consump = 0
            cumulative_t_distill_delay = 0
            cumulative_t_prep_delay = 0
            cumulative_t_inter_handover = 0

            # Initialize `t_intra_consump`, the intra-module consumption time for all required T,Rz-basis
            # measurements per a scheduling step:
            t_intra_consump = 0
            for consump_simultaneous_block in consump_schedule[wid_index]:
                consump_gates = []
                for dictionary in consump_simultaneous_block:
                    consump_gates += list(dictionary.keys())
                t_logicals_per_block = list(set(consump_gates).intersection(t_measures))
                rot_logicals_per_block = list(set(consump_gates).intersection(rot_measures))

                logger.debug(
                    f"For widget:{widget_key}, RRE will now validate basis counts for "
                    f"`consump_simultaneous_block`:{consump_simultaneous_block}.\n"
                )
                if len(t_logicals_per_block) + len(rot_logicals_per_block) > len(measurement_basis_list[wid_index]):
                    raise RuntimeError(
                        "RRE found a mismatch for T,T_dagger,RZ-gate counts in the subsched as follows:\n"
                        f"\tconsump_gates: {consump_gates}\n"
                        f"\tmeasurement_basis_list: {measurement_basis_list[wid_index]}\n"
                        f"\tt_logicals_per_block: {t_logicals_per_block}\n"
                        f"\trot_logicals_per_block: {rot_logicals_per_block}"
                    )
                else:
                    logger.debug(
                        f"For consump_sublist:{consump_gates}, RRE found {len(t_logicals_per_block)} t_basis_measures "
                        f"and {len(rot_logicals_per_block)} rz_basis_measures."
                    )

                # Calculate no. of sequential graph node consump ops required at this consump stage (happening over
                # T-transfer and quantum buses of num_modules_per_leg modules in one leg). We guarantee at worst
                # num_t_factories widgets service parallel consumptions at maximally Delta nodes. Note only distinct
                # Rz's and distinct T's can be consumed in parallel.
                n_max_t = max_nodes_over_modules.n_max_nodes(meas_sublist=t_logicals_per_block)
                n_max_rz = max_nodes_over_modules.n_max_nodes(meas_sublist=rot_logicals_per_block)
                n_seq_consump = t_length_unit * math.ceil(n_max_rz / n2) + math.ceil(n_max_t / n2)

                # Calculate `t_intra_consump`, the intra-module consumption time for all required T,Rz-basis
                # measurments per a scheduling sub-step:
                t_intra_consump += n_seq_consump * 8 * float(self.intra_qcycle_sec) * distance

            # If we require more T states than what was available in T-transfer-bus, we must add delays to
            # `t_intra_consump` to distill more T states. `t_distill_delay` is the intra-module comsumption delay
            # needed to await for distilation of these additianl T states.
            t_distill_delay = 0
            seq_prep_ops = SeqPrepOps(
                prep_subschedules=prep_schedule[wid_index],
                num_modules_per_leg=num_modules_per_leg,
                num_pipes_per_intermodule_connection=num_pipes_per_intermodule_connection,
                delta=delta,
            )
            seq_fridge_crosses = seq_prep_ops.seq_fridge_crosses()  # sequential cross-module ops over prep_schedule
            seq_intra_comms = seq_prep_ops.seq_intra_comms()  # sequential intra-module ops over prep_schedule
            time_intra_prep = (
                (
                    seq_intra_comms * self.intra_qcycle_sec
                    + seq_fridge_crosses * float(self.inter_handover_timescale_sec)
                )
                * 8
                * distance
            )
            # Characteristic time for a T-factory to distill a T-state over C-cycles
            characteristic_t_intra_distill = 8 * self.intra_qcycle_sec * dist_widget.cycles
            n_avail_magicstates = max(
                np.floor(n2 * time_intra_prep / characteristic_t_intra_distill),
                intra_components_counts.len_transfer_bus,
            )
            num_magic_states = len(t_measures) + t_length_unit * len(rot_measures)
            if num_magic_states > n_avail_magicstates:
                extra_seq_distillations = math.ceil((num_magic_states - n_avail_magicstates) / n2)
                t_distill_delay = extra_seq_distillations * 8 * float(self.intra_qcycle_sec) * dist_widget.cycles
                t_intra_consump += t_distill_delay
            logger.debug(
                f"For widget:{widget_key}, we need {num_magic_states} T-states, compared to available "
                f"capacity of {n_avail_magicstates}, leading to a delay of {t_distill_delay} sec.\n"
            )

            # Scaling up `t_intra_consump` according to the algorithm's widget repetitions:
            cumulative_t_intra_consump = t_intra_consump * transpiled_widgets["widgets"][widget_key][1]
            cumulative_t_distill_delay = t_distill_delay * transpiled_widgets["widgets"][widget_key][1]

            # Iterate through the widget tuples in `transpiled_widgets["stitches"]` with their first widget matching
            # `widget_key` to calculate the prep and inter-modular times:
            for widget_tuple in [
                widget_tuple for widget_tuple in transpiled_widgets["stitches"].keys() if widget_tuple[0] == widget_key
            ]:
                logger.debug(f"Calculating cross-module time quantities for widget_tuple={widget_tuple}\n")
                next_index = transpiled_widgets_keys.index(widget_tuple[1])

                # If the next graph prep steps, running in parallel across the other leg of the macro-architecture, are
                # not completed by now, we need to add some idle delays:
                seq_prep_ops = SeqPrepOps(
                    prep_subschedules=prep_schedule[next_index],
                    num_modules_per_leg=num_modules_per_leg,
                    num_pipes_per_intermodule_connection=num_pipes_per_intermodule_connection,
                    delta=delta,
                )
                seq_fridge_crosses = seq_prep_ops.seq_fridge_crosses()
                seq_intra_comms = seq_prep_ops.seq_intra_comms()
                t_prep_next = (
                    (
                        seq_intra_comms * self.intra_qcycle_sec
                        + seq_fridge_crosses * float(self.inter_handover_timescale_sec)
                    )
                    * 8
                    * distance
                )
                if t_prep_next > t_intra_consump:
                    t_prep_delay = t_prep_next - t_intra_consump
                    total_t_prep_delay = t_prep_delay * transpiled_widgets["stitches"][widget_tuple]
                    cumulative_t_intra_consump += total_t_prep_delay
                    cumulative_t_prep_delay += total_t_prep_delay

                # Calculating the upper-bound to the time for stitching input to output nodes: i.e. handing over the
                # data qubits, or the graph quantum state, from one to the corresponding modules in the next leg of
                # ladder architecture. Note graph nodes cannot be teleported vertically for free through O(1) ops.
                t_inter_handover = (
                    math.ceil(
                        max_nodes_over_modules.num_crossmodule_io(
                            output_nodes=output_nodes[wid_index], input_nodes=input_nodes[next_index]
                        )
                        / num_pipes_per_intermodule_connection
                    )
                    * float(self.inter_handover_timescale_sec)
                    * distance
                    * 8
                )
                cumulative_t_inter_handover += t_inter_handover * transpiled_widgets["stitches"][widget_tuple]

            # If `widget_key` matches the very first widget (Step 0), add an initial prep time:
            if transpiled_widgets["first_widget"] == widget_key:
                seq_prep_ops = SeqPrepOps(
                    prep_subschedules=prep_schedule[wid_index],
                    num_modules_per_leg=num_modules_per_leg,
                    num_pipes_per_intermodule_connection=num_pipes_per_intermodule_connection,
                    delta=delta,
                )
                seq_fridge_crosses = seq_prep_ops.seq_fridge_crosses()
                seq_intra_comms = seq_prep_ops.seq_intra_comms()
                total_t_prep_delay = (
                    (
                        seq_intra_comms * self.intra_qcycle_sec
                        + seq_fridge_crosses * float(self.inter_handover_timescale_sec)
                    )
                    * 8
                    * distance
                )
                cumulative_t_intra_consump += total_t_prep_delay
                cumulative_t_prep_delay += total_t_prep_delay

            intra_consump_ops_time_sec += cumulative_t_intra_consump
            intra_distill_delay_sec += cumulative_t_distill_delay
            intra_prep_delay_sec += cumulative_t_prep_delay
            inter_handover_ops_time_sec += cumulative_t_inter_handover
            logging.debug(
                f"For widget_key:{widget_key} of the algorithm, time data are:\n"
                f"intra_consump_ops_time_sec: {intra_consump_ops_time_sec}\n"
                f"intra_distill_delay_sec: {intra_distill_delay_sec}\n"
                f"intra_prep_delay_sec: {intra_prep_delay_sec}\n"
                f"inter_handover_ops_time_sec: {inter_handover_ops_time_sec}\n"
            )

        return ConsumpSchedTimes(
            intra_distill_delay_sec=intra_distill_delay_sec,
            intra_prep_delay_sec=intra_prep_delay_sec,
            intra_consump_ops_time_sec=intra_consump_ops_time_sec,
            inter_handover_ops_time_sec=inter_handover_ops_time_sec,
        )

    def allocate_logicals_to_modules(self, dist_widget: DistWidget, d: int, delta: int) -> int:
        """Check, allocate the routing of qubits, and calculate no. of fridges available per leg for graph processing.

        The allocation is performed based on the ladder-style macroscopic and space-time-optimized intra-module
        architectures given in the RRE manuscript arXiv:2406.06015. There are equal number of cryo-plants in each leg of
        the ladder to prepare and consump subgraphs in an interleaving pattern. The number of fridges per leg is set as
        the minimum required to host the quantum bus and at least one T-factory per module. The intra-module
        architecture has three main components of quantum, T-transfer buses and T-factories. The "unallocated" nodes may
        be still used to aid ancillary operations.

        :param widget: the distillation widget (T-factory) in the form of an DistWidget object
        :param distance: surface code distance
        :param num_logical_qubits: the required logical qubit count. This optionally check if the calculated
            `total_avail_logical_qubits` is larger than this integer.

        :returns: number of fridges assigned to each leg of the macroscopic architecture.
        """
        num_modules_per_leg = 1

        # Initial guesses for the sizes of intra-module components
        intra_component_counts = self.calc_num_t_factories(
            dist_widget=dist_widget,
            d=d,
            delta=math.ceil(delta / num_modules_per_leg),
        )

        while intra_component_counts.num_t_factories == 0:
            num_modules_per_leg += 1
            if num_modules_per_leg > delta:
                raise RuntimeError(
                    f"RRE could not fit a quantum bus of size {delta} in even {delta} number of fridges per leg. "
                    f"Each fridge had a logical size of {intra_component_counts.edge_logical}-by-"
                    f"{intra_component_counts.edge_logical}, was equipped with {intra_component_counts.num_t_factories}"
                    f" T-factories of type {dist_widget.name}, d={d}, and qpu_phys_qubits={self.qpu.num_qubits}.\n"
                )
            else:
                logger.info(
                    f"\nDistributing the quantum bus of size {delta} in num_modules_per_leg={num_modules_per_leg}\n"
                )
            intra_component_counts = self.calc_num_t_factories(
                dist_widget=dist_widget,
                d=d,
                delta=math.ceil(delta / num_modules_per_leg),
            )

        logger.info(
            f"RRE has successfully allocated {intra_component_counts.num_logical_qubits} required logical qubits to "
            f"{num_modules_per_leg} fridges per each leg of the macroscopic architecture. "
            f"Each fridge has a logical size of {intra_component_counts.edge_logical}-by-"
            f"{intra_component_counts.edge_logical}, is equipped with {intra_component_counts.num_t_factories} "
            f"T-factories of type {dist_widget.name}, d={d}, and qpu_phys_qubits={self.qpu.num_qubits}.\n"
        )

        return num_modules_per_leg


def build_system(widget: DistWidget, ftqc_params: dict) -> FTQC:
    """Build FTQC system from values in `params.yaml`.

    :param widget: the distillation widget (T-factory) in the form of an DistWidget object.
    :param ftqc_params: a dictionary of FTQC parameters.

    :returns: a complete FTQC system
    """
    ftqc_params = copy.deepcopy(ftqc_params)
    qpu = QuantumProcessor.from_dict(widget, ftqc_params.pop("processor"))
    fridge = Fridge.from_dict(ftqc_params.pop("fridge"))
    lines_dict = {line["name"]: SignalChain.from_dict(line) for line in ftqc_params.pop("lines")}
    return FTQC(
        name=ftqc_params.pop("name"),
        inter_handover_timescale_sec=ftqc_params.pop("inter_handover_timescale_sec"),
        intra_qcycle_sec=ftqc_params.pop("qcycle_char_timescale_ns") * 1e-9,
        num_pipes_per_intermodule_connection=ftqc_params.pop("num_pipes_per_intermodule_connection"),
        qpu=qpu,
        cryostat=fridge,
        **lines_dict,
    )


def build_dist_widgets(widget_params: dict) -> WidgetTable:
    """Build a distillation widget lookup table from values in `params.yaml`.

    :param widget_params: a dictionary of all widget params

    :returns: the distillation widget lookup table
    """
    table_params = copy.deepcopy(widget_params)
    dist_widget_table = WidgetTable.from_dict(table_params)
    return dist_widget_table
