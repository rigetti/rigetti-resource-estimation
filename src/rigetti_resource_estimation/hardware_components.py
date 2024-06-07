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
**Module** ``hardware_components``

A collection of classes for expressing hardware configurations given a fault-tolerant system architecture.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, NamedTuple
import math
import copy
import logging

import pandas as pd
import numpy as np

from rigetti_resource_estimation.graph_utils import JabalizerSchedGraphItems

logger = logging.getLogger(__name__)


ConsumpSchedTimes = NamedTuple(
    "ConsumpSchedTimes",
    [
        ("intra_consump_ops_time_sec", float),
        ("intra_t_injection_time_sec", float),
        ("inter_handover_ops_time_sec", float),
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

    Holds the definition of fridges such as Bluefors, Bluefors LD, Bluefors XLD.

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
            num_readout_lines = int(num_qubits / qubits_per_readout_line)  # do we need to check for partial lines?
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
    :param num_intermodule_pipes: number of pipes connecting the two fridges.
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
    num_intermodule_pipes: int

    def calc_readout_resources(self):
        """Calculate output parameters related to readout."""
        return self.qpu.num_readout_lines * len(self.readout_out_line.components)

    def calc_thermal_loads(self):
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
                    * 2  # factor 2 comes from the fact that we have a two fridge architecture
                    * num
                )

        return self.cryostat.stages

    def calc_num_t_factories(self, widget_q: int, d: int, req_logical_qubits: int) -> int:
        """Calculate the number of T-factories that fits per fridge after fitting the largest subgraph widget.

        :param widget_q: number of qubits in the widget.
        :param d: code distance.
        :param req_logical_qubits: required logical qubits.
        """
        unalloc_phys_qubits = (
            (self.allocate_logicals_to_modules(distance=d, req_logical_qubits=req_logical_qubits) - req_logical_qubits)
            * 4
            * d**2
        )
        num_t_factories = max(math.floor(unalloc_phys_qubits / widget_q), 0)
        if num_t_factories == 0:
            logger.warning("No T-factory can be fit in the fridges.")
        return num_t_factories

    def calc_consump_sched_time(
        self,
        graph_items: JabalizerSchedGraphItems,
        dist_widget: DistWidget,
        num_intermodule_pipes: int,
        distance: int,
    ) -> ConsumpSchedTimes:
        """Calculate the time quantities relating to the schedules including all intra and inter comminations.

        We assume a bilinear-bus two fridge architecture for the complete fault-tolerant quantum computer by default:
        The architecture includes two interconnected fridge modules hosting algorithmic time-ordered subcircuit widgets
        and T-factories. The two modules run interleaving subgraphs. When a module grows the graph by preparing a new
        subgraph inside, the other in contrast consumes the prior subgraph, which is assumed to always take longer than
        any graph preparation. The quantum bus inside both modules is bilinear and ancilla qubits are laid out in a
        comb-like pattern. Both module host an equal number of T-factory widgets inside, which feed T states internally
        to the fridge that is consuming the subgraph.

        Users may switch to a single-fridge architecture (with T-factories inside) by appropriately configuring
        `inter_handover_timescale_sec`, `qcycle_char_timescale_ns`, and `processor.num_qubits` in `params.yaml`. One can
        also set `num_intermodule_pipes` to speed-up subgraph handover to any desired number.

        :param graph_items: A JabalizerSchedGraphItems object holding the attributes and scheds of the full graph state.
        :param dist_widget: A DistWidget object identifying required distillation widgets (T-factories).
        :param num_intermodule_pipes: number of pipes connecting the two fridges in the architecture.
        :param distance: surface code distance.

        :returns:
            `intra_consump_ops_time_sec`: total intra-module time for surface code ops relating to the consumption of
            the subgraphs in the unit of sec.
            `intra_t_injection_time_sec`: total intra-module time for transferring the T-states to be used for the
            consumption of the subgraph in the same fridge in the unit of sec.
            `inter_handover_ops_time_sec`: total inter-module time for the handover ops stitching the subgraph belonging
            to the current graph state into the subgraph in the second module.
        """
        consump_schedule = graph_items.consump_sched
        prep_schedule = graph_items.prep_sched if graph_items.prep_sched else [[]]
        measurement_basis_list = graph_items.measurement_basis_list if graph_items.measurement_basis_list else [[]]
        output_nodes = graph_items.output_nodes if graph_items.output_nodes else [[]]
        t_length_unit = graph_items.t_length_unit
        widget_q = dist_widget.qubits
        widget_cycles = dist_widget.cycles

        if consump_schedule is None:
            logger.info("The consumption schedule was empty!")
            return ConsumpSchedTimes(
                intra_t_injection_time_sec=0.0,
                intra_consump_ops_time_sec=0.0,
                inter_handover_ops_time_sec=0.0,
            )

        intra_consump_ops_time_sec = 0.0
        intra_t_injection_time_sec = 0.0
        inter_handover_ops_time_sec = 0.0

        for index, consump_subsched in enumerate(consump_schedule):
            logger.debug(
                f"Calculting timescales for step {index} based on the following lists:"
                f"\nconsump (sub)schedule: {consump_subsched}\nmeasurement list: {measurement_basis_list[index]}"
            )

            t_measures = [ind for op, ind, _ in measurement_basis_list[index] if op in ["T", "T_Dagger"]]
            rot_measures = [ind for op, ind, _ in measurement_basis_list[index] if op == "RZ"]

            for consump_gates in consump_subsched:
                t_logicals_per_block = len(set(consump_gates).intersection(t_measures))
                rot_logicals_per_block = len(set(consump_gates).intersection(rot_measures))
                if t_logicals_per_block + rot_logicals_per_block > len(measurement_basis_list[index]):
                    raise RuntimeError(
                        "RRE found a mismatch for T,T_Dagger,RZ-gate count as per following lists:\n"
                        f"consump_gates: {consump_gates}\n"
                        f"measurement_basis_list: {measurement_basis_list[index]}\n"
                        f"t_logicals_per_block: {t_logicals_per_block}\n"
                        f"rot_logicals_per_block: {rot_logicals_per_block}"
                    )
                else:
                    logger.debug(
                        f"For consump_gates:{consump_gates}, RRE found {t_logicals_per_block} t_basis_measures "
                        f"and {rot_logicals_per_block} rot_basis_measures."
                    )

            # Calculating number of nonparallel surface code cycles at each stage
            n_factories = self.calc_num_t_factories(
                widget_q=widget_q, d=distance, req_logical_qubits=int(graph_items.delta or 0)
            )
            if n_factories == 0:
                return ConsumpSchedTimes(
                    intra_t_injection_time_sec=0,
                    intra_consump_ops_time_sec=0,
                    inter_handover_ops_time_sec=0,
                )
            reps_consump = t_length_unit * math.ceil(len(rot_measures) / n_factories) + math.ceil(
                len(t_measures) / n_factories
            )  # Distinct Rz's & T's can be consumed in parallel
            reps_magicstates = math.ceil((len(t_measures) + t_length_unit * len(rot_measures)) / n_factories)

            t_intra_consump = reps_consump * 8 * float(self.intra_qcycle_sec) * distance
            t_intra_magicstates = reps_magicstates * 8 * (widget_cycles + distance) * float(self.intra_qcycle_sec)

            t_prep_next = (
                0
                if index == len(prep_schedule) - 1
                else len(prep_schedule[index + 1]) * 8 * float(self.intra_qcycle_sec) * distance
            )
            if t_prep_next > t_intra_consump + t_intra_magicstates:
                logger.info(
                    f"Step {index} of schedules: the architectural assumption of "
                    "t_prep_next > t_intra_consump+t_intra_magicstates is invalid. Therefore we manually set a delayed"
                    "t_intra = t_prep_next - t_inter."
                )
                t_intra_consump = t_prep_next - t_intra_magicstates

            t_inter_handover = (
                math.ceil(len(output_nodes[index]) / num_intermodule_pipes)
                * float(self.inter_handover_timescale_sec)
                * distance
                if index < len(consump_schedule) - 1
                else 0
            )

            intra_consump_ops_time_sec += t_intra_consump
            intra_t_injection_time_sec += t_intra_magicstates
            inter_handover_ops_time_sec += t_inter_handover
            logging.debug(
                f"Step {index} of schedules: we validated t_intra+t_inter:{t_intra_consump+t_intra_magicstates} >= "
                f"t_prep_next:{t_prep_next} sec -- currently, we have "
                f"inter_consump_ops_time_sec:{intra_t_injection_time_sec}, "
                f"intra_consump_ops_time_sec={intra_consump_ops_time_sec}, and "
                f"inter_handover_ops_time_sec={inter_handover_ops_time_sec}."
            )

        return ConsumpSchedTimes(
            intra_t_injection_time_sec=intra_t_injection_time_sec,
            intra_consump_ops_time_sec=intra_consump_ops_time_sec,
            inter_handover_ops_time_sec=inter_handover_ops_time_sec,
        )

    def logical_number_in_rect_comb(self, rect: List) -> int:
        """Find the number of graph nodes per rail of bilinear bus available in a rectangular array of logical qubits.

        :param rect: A list of dimensions of the rectangular array in terms of the logical qubits.

        :returns: number of logical qubits available.
        """
        rect.sort()  # we want the long edge to efficiently layout qubits
        height = rect[0]
        height = height - 1 if (height % 2) != 0 else height  # discard odd rows for easier layout
        width = rect[1]

        if width < 3:
            return 0  # can't fit a comb of ancillae

        if height < 2:
            return 0  # can't fit any ancillae
        elif height == 2:
            return width  # for 2 rows, half are logical qubits
        elif height > 2:
            return int(width * height / 2 - height / 2)

        return 0

    def allocate_logicals_to_modules(self, distance: int, req_logical_qubits: int = 1) -> int:
        """Check and allocate the routing of logical qubits for processing the (sub)graph states.

        Note that all qubits that are not allocated for state distillation, are used for processing the graph state:
        encoding the algorithm in logical qubits or as ancillae for communicating between them.

        :param widget: the distillation widget (T-factory) in the form of an DistWidget object
        :param distance: surface code distance

        :returns: maximum number of logical qubits available per module
        """
        logical_edge_len = np.ceil(np.sqrt(2) * distance)
        num_logicals = np.floor(np.sqrt(self.qpu.num_qubits) / logical_edge_len)
        avail_logical_qubits = self.logical_number_in_rect_comb([num_logicals, num_logicals])
        if req_logical_qubits >= avail_logical_qubits:
            logger.warning(
                f"It is impossible to fit {req_logical_qubits} required logical qubits, extracted from the largest "
                f"subcircuit, in the fridges of logical sizes {int(num_logicals)}-by-{int(num_logicals)}, d={distance},"
                f" and no_phys_qubits={self.qpu.num_qubits}.\n"
            )
        return avail_logical_qubits


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
        num_intermodule_pipes=ftqc_params.pop("num_intermodule_pipes"),
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
