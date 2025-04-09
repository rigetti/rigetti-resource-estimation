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


def find_physq_per_patch(distance: int) -> int:
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
        self, name: str = "Code distance", short_name: str = "distance", description: str = "Surface code distance, d"
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated value of this resource.

        :param experiment: experiment to calculate the distance for.
        """
        return experiment.distance


class NumLogicalQubits(Resource):
    """
    Resource class for the number of logical qubits per rail of bus or max quantum memory required at any given time.

    The max quantum memory is extracted from the space requirements of the largest subgraph consumption schedule. We
    report the logical patches required in each rail of the bilinear quantum bus distributed over a ladder leg. We
    ignore the logical patches required for the T-transfer bus and distillation widgets. This resource can be considered
    the main identifier for the space costs.
    """

    def __init__(
        self,
        name: str = "Number of logical qubits per bus rail",
        short_name: str = "num_logical_qubits_per_busrail",
        description: str = "Number of logical qubits per quantum bus rail (max memory or graph delta)",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated required number of logical qubits.

        :param experiment: experiment to extract required attributes of the architecture.
        """
        return experiment.graph_items.delta or 0


class NumTFactories(Resource):
    """Number of T-distilleries (MSD factories) in each module."""

    def __init__(
        self,
        name: str = "Number of T-factories per module",
        short_name: str = "num_t_factories_per_module",
        description: str = "Number of T-factories per module",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return total number of T-factories present in each module.

        :param experiment: experiment to calculate the total number of T-factories.
        """
        return experiment.intra_component_counts.num_t_factories


class MemoryLogicalQubits(Resource):
    """
    A resource class for the number of logical memory (graph-state) qubits in the bilinear bus portion per module.

    All modules on a leg of the macro-architecture host the same number of memory logical qubits
    equals to `memory_logical_qubits_per_module`. The last module may need less than `memory_logical_qubits_per_module`
    nodes for quantum operations; however, the same space is always reserved for this
    component.
    """

    def __init__(
        self,
        name: str = "Logical memory logical qubits per module",
        short_name: str = "memory_logical_qubits_per_module",
        description: str = "Number of logical memory (graph-state) qubits in the bilinear bus per module",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated required memory logical qubits per module.

        :param experiment: experiment to extract required attributes of the architecture.
        """
        return math.ceil((experiment.graph_items.delta or 0) / experiment.num_modules_per_leg)


class MemoryPhysicalQubits(Resource):
    """A resource class for the number of physical memory (graph-state) qubits in the bilinear bus per module."""

    def __init__(
        self,
        memory_logical_qubits: MemoryLogicalQubits,
        name: str = "Physical memory qubits per module",
        short_name: str = "memory_physical_qubits_per_module",
        description: str = "Number of memory physical (graph-state) qubits in the bilinear bus per module",
    ) -> None:
        super().__init__(name, short_name, description)
        self.memory_logical_qubits = memory_logical_qubits

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated required memory physical qubits per module.

        :param experiment: experiment to extract required attributes of the architecture.
        """
        return self.memory_logical_qubits.get_value(experiment) * find_physq_per_patch(experiment.distance)


class AncillaLogicalQubits(Resource):
    """
    A resource class for the number of logical ancilla qubits in the bilinear quantum bus section per module.

    Ancilla qubits facilitate all logical operations through multipartite measurements. All modules on a leg of the
    macro-architecture host the same number of ancilla logical qubits equals to `ancilla_logical_qubits_per_module`.
    The last module may need less than `ancilla_logical_qubits_per_module` nodes for quantum operations; however, the
    same space is always reserved for this component.
    """

    def __init__(
        self,
        name: str = "Logical ancilla qubits per module",
        short_name: str = "ancilla_logical_qubits_per_module",
        description: str = "Number of logical ancilla qubits in the bilinear bus per module",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated required logical memory qubits per module.

        :param experiment: experiment to extract required attributes of the architecture.
        """
        return math.ceil((experiment.graph_items.delta or 0) / experiment.num_modules_per_leg)


class AncillaPhysicalQubits(Resource):
    """A resource class for the number of physical ancilla qubits in the bilinear bus section per module."""

    def __init__(
        self,
        ancilla_logical_qubits: AncillaLogicalQubits,
        name: str = "Physical ancilla qubits per module",
        short_name: str = "ancilla_physical_qubits_per_module",
        description: str = "Number of physical ancilla qubits in the bilinear bus per module",
    ) -> None:
        super().__init__(name, short_name, description)
        self.ancilla_logical_qubits = ancilla_logical_qubits

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated required physical ancilla qubits per module.

        :param experiment: experiment to extract required attributes of the architecture.
        """
        return self.ancilla_logical_qubits.get_value(experiment) * find_physq_per_patch(experiment.distance)


class TBufferLogicalQubits(Resource):
    """A resource class for the number of logical qubits in the T-transfer bus per module."""

    def __init__(
        self,
        name: str = "Logical T-buffer qubits per module",
        short_name: str = "t_buffer_logical_qubits_per_module",
        description: str = "Number of logical qubits in the T-transfer bus per module",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated logical T-buffer qubits per module.

        :param experiment: experiment to extract required attributes of the architecture.
        """
        return experiment.intra_component_counts.len_transfer_bus


class TBufferPhysicalQubits(Resource):
    """A resource class for the number of physical qubits in the T-transfer bus per module."""

    def __init__(
        self,
        name: str = "Physical T-buffer qubits per module",
        short_name: str = "t_buffer_physical_qubits_per_module",
        description: str = "Number of physical qubits in the T-transfer bus per module",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated required physical T-buffer qubits per module.

        :param experiment: experiment to extract required attributes of the architecture.
        """
        return experiment.intra_component_counts.len_transfer_bus * find_physq_per_patch(experiment.distance)


class DistillWidgetLogicalQubits(Resource):
    """Number of logical nodes assigned to all T-distillation widgets in each module."""

    def __init__(
        self,
        name: str = "Logical T-distillery qubits per module",
        short_name: str = "t_distillery_logical_qubits_per_module",
        description: str = "Number of logical nodes assigned to all T-distillation widgets in each module",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated value of Widget Qubits.

        :param experiment: experiment to calculate the widget logical qubits value for.
        """
        widget_logical_l = math.ceil(experiment.widget.length / (math.sqrt(2) * experiment.distance))
        widget_logical_w = math.ceil(experiment.widget.width / (math.sqrt(2) * experiment.distance))
        return widget_logical_w * widget_logical_l * experiment.intra_component_counts.num_t_factories


class DistillWidgetPhysicalQubits(Resource):
    """Number of active physical qubits in all T-distillation widgets in each module."""

    def __init__(
        self,
        name: str = "T-distillery physical qubits per module",
        short_name: str = "t_distillery_physical_qubits_per_module",
        description: str = "Number of active physical qubits in all T-distilleries in each module",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated value of Widget Qubits.

        :param experiment: experiment to calculate the widget qubits value for.
        """
        return int(experiment.widget.qubits * experiment.intra_component_counts.num_t_factories)


class TotalNumFridges(Resource):
    """Calculate total number of QPU cryo-modules on both leg of the ladder macro-architecture."""

    def __init__(
        self,
        name: str = "Total number of cryo-modules",
        short_name: str = "total_num_modules",
        description: str = "Total number of modules on both legs of macro-architecture",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated total number of modules.


        :param experiment: experiment to calculate the total number of modules.
        """
        return 2 * experiment.num_modules_per_leg


class NumIntermoduleConnections(Resource):
    """Calculate total number of interconnects in the ladder macro-architecture"""

    def __init__(
        self,
        name: str = "Total number of interconnects",
        short_name: str = "total_num_intermodule_connections",
        description: str = "Total number of interconnects in the ladder macro-architecture",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated total no. of interconnects.

        :param experiment: experiment to calculate the total number of interconnects.
        """
        return (
            3 * experiment.num_modules_per_leg - 2
        ) * experiment.system_architecture.num_pipes_per_intermodule_connection


class NumPhysicalQubits(Resource):
    """
    Calculate the number of allocated physical qubits in a module of the FTQC at any given time.

    The number if allocated physical qubits, `total_num_physical_qubits`, is always smaller or equal to the fixed total
    number of available physical qubits. We perform a component allocation trying to minimize the number of
    `unallocated physical qubits`, which sets the requited and fixed total physical count of the module as close as
    possible. For details of the intra-module fault-tolerant architecture with T-state parallelization and explicit
    component allocation see `hardware_components.py` and the RRE manuscript arXiv:2406.06015.

    Users may switch to a single-fridge architecture (with internal T-factories) by appropriately configuring
    `inter_handover_timescale_sec`, `qcycle_char_timescale_ns`, and `processor.num_qubits` in `params.yaml`. One can
    also set `num_pipes_per_intermodule_connection` to speed-up subgraph handover to any desired rate.
    """

    def __init__(
        self,
        name: str = "Number of allocated physical qubits per module",
        short_name: str = "num_alloc_physical_qubits_per_module",
        description: str = "Number of allocated physical qubits per module",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return calculated value of physical qubits.


        :param experiment: experiment to calculate the required number of qubits per module.
        """
        physq_per_patch = find_physq_per_patch(experiment.distance)
        return int(
            math.ceil((experiment.graph_items.delta or 0) / experiment.num_modules_per_leg) * 2 * physq_per_patch
            + (
                +experiment.intra_component_counts.num_t_factories * experiment.widget.qubits
                + experiment.intra_component_counts.len_transfer_bus * physq_per_patch
            )
        )


class InputLogicalQubits(Resource):
    """
    Number of logical qubits or width of the complete input algorithm.

    For the time-sliced widgetization we perform on the input algorithm, all widgets have the same number of output and
    input nodes equal to this number.
    """

    def __init__(
        self,
        name: str = "Input logical qubits",
        short_name: str = "input_logical_qubits",
        description: str = "Number of algorithmic logical qubits",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of logical qubits in the input circuit.

        :param experiment: experiment to calculate the number of algorithmic logical qubits for.
        """
        return experiment.input_logical_qubits


class DiamondNormEps(Resource):
    """Diamond norm epsilon-value for gate-synth decomposition of arbitrary Rz's to Clifford+T."""

    def __init__(
        self,
        name: str = "Diamond norm epsilon",
        short_name: str = "diamond_norm_eps",
        description: str = "Diamond-norm epsilon for gate-synth decompositions, eps",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> float:
        """Return the diamond norm epsilon.

        :param experiment: experiment to calculate the diamond norm epsilon for.
        """
        return experiment.diamond_norm_eps


class TCount(Resource):
    """Total T-count of input circuit after performing all the required gate-synth operations."""

    def __init__(
        self,
        name: str = "T-count",
        short_name: str = "t_count",
        description: str = "Total number of T-basis measurements required at the consumption stage, T-count",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the circuit's T-Count.

        :param experiment: experiment to calculate the T-Count for.
        """
        return experiment.graph_items.t_count()


class TDepth(Resource):
    """Estimates effective T-depth of the input circuit after performing all the required gate-synth operations."""

    def __init__(
        self,
        name: str = "Circuit T-depth",
        short_name: str = "t_depth",
        description: str = "Circuit's effective T-depth",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the circuit's T depth.

        :param experiment: experiment to calculate the T depth for.
        """
        delta = experiment.graph_items.delta or 0
        if delta == 0:
            raise RuntimeError(f"Captured delta={delta}, making T-depth calculations infeasible.")
        else:
            return math.ceil(experiment.graph_items.t_count() / delta)


class InitRZCount(Resource):
    """
    The number of arbitrary (non-Clifford) angle RZ gates in the input circuit.

    The gates can be written explicitly or contained within other logical gates. All Rz gates must undergo
    gate-synthesis decomposition to Clifford+T at the end, during the consumption stage.
    """

    def __init__(
        self,
        name: str = "RZ count",
        short_name: str = "rz_count",
        description: str = "Number of arbitrary-rotation Rz gates in the input logical circuit, Rz-count",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the circuit's RZ count.

        :param experiment: experiment to calculate the RZ count for.
        """
        return experiment.graph_items.rz_count or 0


class InitTCount(Resource):
    """
    The number of logical T and TDagger gates in the complete input circuit.

    The gates can be written explicitly or contained within other logical gates. We perform this count
    before performing gate-synthesis Clifford+T decomposition of arbitrary-angle Rz-gates.
    """

    def __init__(
        self,
        name: str = "Initial T-count",
        short_name: str = "init_t_count",
        description: str = "Number of logical T,TDagger-gates in the initial circuit before performing gate-synth",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the circuit's initial T count.

        :param experiment: experiment to calculate the initial T count for.
        """
        return experiment.graph_items.t_count_init or 0


class InitCliffordCount(Resource):
    """Number of explicit logical Clifford gates in the complete input circuit."""

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
    """A resource class for the approximate size of the complete graph state."""

    def __init__(
        self, name: str = "Graph size", short_name: str = "N", description: str = "Number of graph nodes, N"
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the graph size.

        :param experiment: experiment to calculate the graph size for.
        """
        return experiment.graph_items.big_n or 0


class ConsumpScheduleSteps(Resource):
    """Total number of sequential measurement steps in consumption schedule over all widgets (repeated or not)."""

    def __init__(
        self,
        name: str = "Consumption schedule size",
        short_name: str = "consumption_schedule_size",
        description: str = "Total number of consumption schedule measurement steps over all widgets.",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return total number of graph consumption measurement steps over all widgets.

        :param experiment: experiment to count the graphs schedule steps.
        """
        return experiment.graph_items.big_s_consump or 0


class PrepScheduleSteps(Resource):
    """Total number of preparation schedule measurement steps over all widgets (repeated or not)."""

    def __init__(
        self,
        name: str = "Preparation schedule size",
        short_name: str = "preparation_schedule_size",
        description: str = "Total number of preparation schedule measurement steps over all widgets",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return total number of graph preparation measurement steps over all widgets.

        :param experiment: experiment to count the graph preparation schedule steps.
        """
        return experiment.graph_items.big_s_prep or 0


class NumWidgetSteps(Resource):
    """
    Total number of widget repetitions or time steps in the input algorithm.

    If this is larger than one, a widgetization by segmenting the algorithm in the time-direction was performed. This
    parameter can become larger than `num_distinct_widgets` as many repetition of a specific widget may exist.
    """

    def __init__(
        self,
        name: str = "Total number of widget steps",
        short_name: str = "num_widget_steps",
        description: str = "Total number of widget repetitions or widgetization time steps",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of widgetization time steps.

        :param experiment: experiment to calculate the total number of widgets for.
        """
        return sum([value[1] for value in experiment.transpiled_widgets["widgets"].values()])


class NumDistinctWidgets(Resource):
    """
    Number of distinct widgets in the input algorithm.

    This parameter identifies the number of widgets that FTQC needs to compile.
    """

    def __init__(
        self,
        name: str = "Number of distinct widgets",
        short_name: str = "num_distinct_widgets",
        description: str = "Number of distinct widgets",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of distinct widgets.

        :param experiment: experiment to calculate the number of widgets for.
        """
        return len(experiment.transpiled_widgets["widgets"].keys())


class DecoderTock(Resource):
    """Decoder tock in seconds."""

    def __init__(
        self,
        name: str = "Decoding tock",
        short_name: str = "decoder_tock",
        description: str = "Decoder tock in seconds",
        unit="second",
    ) -> None:
        super().__init__(name, short_name, description, unit)

    def get_value(self, experiment: Experiment) -> float:
        """Return the decoder tock time in seconds.

        :param experiment: experiment to calculate the decoder tock time for.
        """
        return experiment.distance * experiment.decoder_char_timescale_sec


class QuantumIntraTock(Resource):
    """Intra-modular tock (seconds) for lattice surgery and consumption stage operations."""

    def __init__(
        self,
        name: str = "Quantum Tock",
        short_name: str = "quantum_tock",
        description: str = "Quantum intra-modular tock for graph processing (seconds)",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> float:
        """Return the tock time for all intra-module quantum operations.

        :param experiment: experiment to calculate the quantum tock time for.
        """
        return 8 * experiment.distance * experiment.system_architecture.intra_qcycle_sec


class TStateTock(Resource):
    """Intra-modular tock (seconds) to distill a purified T-state required for graph consumption."""

    def __init__(
        self,
        name: str = "T-state distillation tock",
        short_name: str = "t_state_tock",
        description: str = "Intra-module tock for T-state distillations (seconds)",
        unit: str = "second",
    ) -> None:
        super().__init__(name, short_name, description, unit)

    def get_value(self, experiment: Experiment) -> float:
        """Return the T state tock time.

        :param experiment: experiment to calculate the T state tock time for.
        """
        return 8 * experiment.widget.cycles * experiment.system_architecture.intra_qcycle_sec


class AvailPhysicalQubits(Resource):
    """Total number of physical qubits available across all modules of the FTQC."""

    def __init__(
        self,
        name: str = "Total number of available physical qubits",
        short_name: str = "total_avail_physical_qubits",
        description: str = "Total number of available physical qubits",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of physical qubits available in both modules.

        :param experiment: experiment to calculate the available physical qubits for.
        """
        return 2 * experiment.num_modules_per_leg * experiment.system_architecture.qpu.num_qubits


class AvailLogicalQubits(Resource):
    """The number of logical qubits available per module of the FTQC."""

    def __init__(
        self,
        name: str = "Available logical qubits per module",
        short_name: str = "avail_logical_qubits_per_module",
        description: str = "Number of available logical qubits per module",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of available logical qubits.

        :param experiment: experiment to calculate the available logical qubits for.
        """
        return experiment.intra_component_counts.edge_logical**2


class UnallocLogicalQubits(Resource):
    """Total number of leftover (unallocated) logical qubits considering all modules and operations of the FTQC."""

    def __init__(
        self,
        name: str = "Total number of unallocated logical qubits",
        short_name: str = "total_unallocated_logical_qubits",
        description: str = "Total number of unallocated logical qubits considering all modules and operations",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of unallocated qubits per module.

        :param experiment: experiment to count the unallocated logical qubits.
        """
        return 2 * int(experiment.intra_component_counts.unalloc_logical_qubits) * experiment.num_modules_per_leg


class UnallocPhysicalQubits(Resource):
    """Total number of unallocated physical qubits considering all modules and operations of the FTQC."""

    def __init__(
        self,
        unalloc_logical_qubits: UnallocLogicalQubits,
        name: str = "Total unallocated physical qubits",
        short_name: str = "total_unallocated_physical_qubits",
        description: str = "Total number of unallocated physical qubits considering all modules and operations",
    ) -> None:
        super().__init__(name, short_name, description)
        self.unalloc_logical_qubits = unalloc_logical_qubits

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of unallocated physical qubits per module.

        :param experiment: experiment to count the unallocated physical qubits for.
        """
        return self.unalloc_logical_qubits.get_value(experiment) * find_physq_per_patch(experiment.distance)


class AllocLogicalQubits(Resource):
    """Total number of allocated logical qubits considering all modules and operations of the FTQC."""

    def __init__(
        self,
        name: str = "Total allocated logical qubits",
        short_name: str = "total_allocated_logical_qubits",
        description: str = "Total number of allocated logical qubits considering all modules and operations",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of allocated qubits per module.

        :param experiment: experiment to calculate the allocated logical qubits for.
        """
        return (
            2
            * (
                experiment.intra_component_counts.edge_logical**2
                - int(experiment.intra_component_counts.unalloc_logical_qubits)
            )
            * experiment.num_modules_per_leg
        )


class AllocPhysicalQubits(Resource):
    """Total number of allocated physical qubits considering all modules and operations of the FTQC."""

    def __init__(
        self,
        alloc_logical_qubits: AllocLogicalQubits,
        name: str = "Total allocated physical qubits",
        short_name: str = "total_allocated_physical_qubits",
        description: str = "Total number of allocated physical qubits considering all modules and operations",
    ) -> None:
        super().__init__(name, short_name, description)
        self.alloc_logical_qubits = alloc_logical_qubits

    def get_value(self, experiment: Experiment) -> int:
        """Return the number of unallocated physical qubits per module.

        :param experiment: experiment to count the unallocated physical qubits.
        """
        return self.alloc_logical_qubits.get_value(experiment) * find_physq_per_patch(experiment.distance)


class ConsumpStageDecodingCores(Resource):
    """The maximum number of concurrent decoding cores running at the consumption stages"""

    def __init__(
        self,
        decoder_tock: DecoderTock,
        name: str = "Number of consumption stage decoding cores",
        short_name: str = "num_consump_concurrent_cores_decoding",
        description: str = "Number of concurrent decoding cores running at consumption stages",
    ) -> None:
        """
        :param decoder_tock: decoder tock (secs).
        """
        super().__init__(name, short_name, description)
        self.decoder_tock = decoder_tock

    def get_value(self, experiment: Experiment) -> int:
        """Return number of decoding cores required at the consumption stage.

        :param experiment: experiment to count the decoder cores.
        """
        q_tock = 8 * experiment.distance * experiment.system_architecture.intra_qcycle_sec
        return math.ceil(self.decoder_tock.get_value(experiment) / q_tock)


class ChipArea(Resource):
    """Total area required for all modules of the QPU."""

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
        return experiment.system_architecture.qpu.processor_area_sqmm * experiment.num_modules_per_leg * 1e-6 * 2


class NumberCouplers(Resource):
    """The total number of required couplers."""

    def __init__(
        self,
        name: str = "Total number of couplers",
        short_name: str = "total_number_of_couplers",
        description: str = "Total number of couplers",
    ) -> None:
        super().__init__(name, short_name, description)

    def get_value(self, experiment: Experiment) -> int:
        """Return number of required couplers.

        :param experiment: experiment to calculate the number of couplers for.
        """
        return experiment.system_architecture.qpu.num_couplers * 2 * experiment.num_modules_per_leg


class DecodingConsumptionPower(Resource):
    """Decoding power used at consumption stage.

    Based on a 100W reference decoding core (aligned with modern GPU or FPGA-based units).
    """

    def __init__(
        self,
        consump_cores: ConsumpStageDecodingCores,
        name: str = "Consumption Stage Decoding Power",
        short_name: str = "consump_decoding_power_kW",
        description: str = "Decoding power at consumption stage (kW)",
        unit="kW",
    ) -> None:
        """
        :param consump_cores: number of concurrent decoding cores used at each consumption step.
        """
        super().__init__(name, short_name, description, unit)
        self.consump_cores = consump_cores

    def get_value(self, experiment: Experiment) -> float:
        """Return decoding power used during consumption stage.

        :param experiment: experiment to calculate the decoding power for.
        """
        ref_pow_watt = 100
        return self.consump_cores.get_value(experiment) * ref_pow_watt * 0.001


class PowerDissipation(Resource):
    """Power dissipation at 4K stage."""

    def __init__(
        self,
        thermal_loads_index: int = 0,
        name: str = "Power dissipation at 4K stage",
        short_name: str = "power_dissip_4K_kW",
        unit: str = "kW",
        description: str = "Total power dissipation at 4K stage (kW)",
    ) -> None:
        """
        :param thermal_loads_index: integer to specify which thermal load to use. 4K: tl_index=0, or mxc: tl_index=1
        """
        super().__init__(name, short_name, description, unit)
        self.tl_index = thermal_loads_index

    def get_value(self, experiment: Experiment) -> float:
        """Return power dissipation at the 4K (tl_index=0) or MXC (tl_index=1) stages.

        :param experiment: experiment to calculate the power dissipation for.
        """
        therm_loads = experiment.system_architecture.calc_thermal_loads(experiment.num_modules_per_leg)
        return (therm_loads[self.tl_index].power_dissipation_watt or 0) * 0.001


class TotalConsumpOpsTime(Resource):
    """Total time required for consumption ops including delays from prep and T-distillations, excluding hand-overs."""

    def __init__(
        self,
        name: str = "Total graph consump ops time",
        short_name: str = "total_consump_ops_sec",
        description: str = "Total graph consumption operations time (sec)",
        unit="sec",
    ) -> None:
        super().__init__(name, short_name, description, unit)

    def get_value(self, experiment: Experiment) -> float:
        """Return the total time taken for graph consumption ops.

        :param experiment: experiment to calculate the total graph consumption time for.
        """
        intra_comps_counts = experiment.intra_component_counts
        delta = experiment.graph_items.delta or 0
        num_modules = experiment.num_modules_per_leg
        if experiment.graph_items.t_counting_cond:
            n_row_qbus = max(math.floor((math.ceil(delta / num_modules) + 1) / intra_comps_counts.l_qbus), 1)
            n2 = min(
                (
                    intra_comps_counts.num_t_factories * 4
                    if experiment.widget.cond_20to4
                    else intra_comps_counts.num_t_factories
                ),
                n_row_qbus,
            )
            len_t_measures = experiment.graph_items.t_count_init
            len_rot_measures = experiment.graph_items.rz_count
            t_length_unit = experiment.graph_items.t_length_unit
            total_seqs_distill = math.ceil((len_t_measures + t_length_unit * len_rot_measures) / n2)
            total_seqs_consump = t_length_unit * math.ceil(len_rot_measures / n2) + math.ceil(len_t_measures / n2)
            # We assume the graph prep times are negligible for the T-counting case. Additionally, we assume graph
            # consumption and T-distillations are performed sequentially.
            intra_consump_ops_time_sec = (
                (total_seqs_distill * experiment.widget.cycles + total_seqs_consump * experiment.distance)
                * 8
                * experiment.system_architecture.intra_qcycle_sec
            )
        else:
            intra_consump_ops_time_sec = experiment.consump_sched_times.intra_consump_ops_time_sec

        return intra_consump_ops_time_sec


class TotalHandoverTime(Resource):
    """Total subgraph handover time to teleport all output to input nodes on the next leg of the macro-architecture."""

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


class OverallTDistillDelay(Resource):
    """Overall delay added to the total consumption time to distill additional T-states."""

    def __init__(
        self,
        name: str = "Total delay for T-state distillations",
        short_name: str = "total_t_distill_delay_sec",
        description: str = "Total T-state distillation delay (sec)",
        unit: str = "sec",
    ) -> None:
        super().__init__(name, short_name, description, unit)

    def get_value(self, experiment: Experiment) -> float:
        """Return the overall delay required to distill addition magic states.

        :param experiment: experiment to calculate the total T-state delay for.
        """
        return experiment.consump_sched_times.intra_distill_delay_sec


class OverallPrepDelay(Resource):
    """Overall delay added to the total consumption time to prepare the next subgraphs in the schedule."""

    def __init__(
        self,
        name: str = "Total delay for graph state preparation",
        short_name: str = "total_graph_prep_delay_sec",
        description: str = "Total graph preparation delay (sec)",
        unit: str = "sec",
    ) -> None:
        super().__init__(name, short_name, description, unit)

    def get_value(self, experiment: Experiment) -> float:
        """Return the graph prep delay time.

        :param experiment: experiment to calculate the total preparation delay.
        """
        return experiment.consump_sched_times.intra_prep_delay_sec


class OverallDecodingDelay(Resource):
    """
    Overall decoding delay in sec.

    Here, we assume a staggered architecture where there are enough multiple reference digital units to processes
    overlapping decoding tasks for a single widget. Therefore per-widget wall-time is only delayed by an overall
    decoding delay.
    """

    def __init__(
        self,
        decoder_tock: DecoderTock,
        total_consump_ops_time: TotalConsumpOpsTime,
        overall_t_state_delay: OverallTDistillDelay,
        name: str = "Overall decoding delay",
        short_name: str = "overall_decoding_delay_sec",
        description: str = "Overall decoding delay (fully classical, sec)",
        unit: str = "sec",
    ) -> None:
        """
        :param decoder_tock: decoder tock.
        :param total_intra_qops_time: Total time for algorithmic surface code ops and parity meas for graph consumption.
        :param overall_t_state_delay: Total T state distillation and injection time.
        """
        super().__init__(name, short_name, description, unit)
        self.decoder_tock = decoder_tock
        self.total_qops_time = total_consump_ops_time
        self.overall_distill_delay = overall_t_state_delay

    def get_value(
        self,
        experiment: Experiment,
    ) -> float:
        """Return the overall decoding delay.

        :param experiment: experiment to calculate the overall decoding delay for.
        """
        dec_tock = self.decoder_tock.get_value(experiment)
        q_tock = 8 * experiment.distance * experiment.system_architecture.intra_qcycle_sec
        magicstate_tock = 8 * experiment.widget.cycles * experiment.system_architecture.intra_qcycle_sec
        total_qops_time = self.total_qops_time.get_value(experiment) - self.overall_distill_delay.get_value(experiment)
        total_decoding_delay_sec = 0.0
        reps_consump = math.ceil(total_qops_time / q_tock)
        reps_magicstate = math.ceil(self.overall_distill_delay.get_value(experiment) / magicstate_tock)
        if dec_tock > q_tock:  # If dec_tock is longer, we need to add the delay for each consumption cycle
            total_decoding_delay_sec += reps_consump * (dec_tock - q_tock)
        if dec_tock > magicstate_tock:  # If dec_tock is longer, we need to add the delay for each distill-inject cycle
            total_decoding_delay_sec += reps_magicstate * (dec_tock - magicstate_tock)
        return total_decoding_delay_sec


class FTWallTime(Resource):
    """Fault-tolerant wall-time considering all non-simultaneous ops of the FTQC over a single algorithmic step."""

    def __init__(
        self,
        total_consump_ops_time: TotalConsumpOpsTime,
        total_handover_time: TotalHandoverTime,
        overall_decoding_delay: OverallDecodingDelay,
        name: str = "Fault-tolerant wall-time",
        short_name: str = "algorithm_step_ft_time_sec",
        description: str = "Fault-tolerant wall-time over a single algorithmic step (upper bound in seconds)",
        unit="sec",
    ) -> None:
        """
        :param total_intra_qops_time: total time for algorithmic surface code ops and parity meas for graph consumption.
        :param total_handover_time: total subgraph handover time.
        :param dec_delay: total time delay for decoding.
        :param total_t_state_time: total T state distillation and injection time.
        """
        super().__init__(name, short_name, description, unit)
        self.total_intra_qops_time = total_consump_ops_time
        self.dec_delay = overall_decoding_delay
        self.total_handover_time = total_handover_time

    def get_value(
        self,
        experiment: Experiment,
    ) -> float:
        """Return the fault-tolerant wall-time.

        :param experiment: experiment to calculate the fault-tolerant wall-time for.
        """
        return (
            self.total_intra_qops_time.get_value(experiment)
            + self.total_handover_time.get_value(experiment)
            + self.dec_delay.get_value(experiment)
        )


class TotalFTAlgorithmTime(Resource):
    """Total fault-tolerant hardware time over all algorithmic steps (the full algorithm time)."""

    def __init__(
        self,
        ft_time: FTWallTime,
        name: str = "Total fault-tolerant algorithm time",
        short_name: str = "total_ft_algorithm_time_sec",
        description: str = "Total fault-tolerant algorithm time (sec)",
        unit: str = "sec",
    ) -> None:
        """
        :param ft_time: fault-tolerant hardware time over a single time step
        """
        super().__init__(name, short_name, description, unit)
        self.ft_time = ft_time

    def get_value(self, experiment: Experiment) -> float:
        """Return the total fault-tolerant time.

        :param experiment: experiment to calculate the total fault-tolerant time for.
        """
        return experiment.num_algorithm_steps * self.ft_time.get_value(experiment)


class TotalFTEnergy(Resource):
    """Upper bound to the total consumed energy accross all modules, operations, and algorithmic steps."""

    def __init__(
        self,
        consump_concurrent_cores: ConsumpStageDecodingCores,
        total_ft_time: TotalFTAlgorithmTime,
        power_4k: PowerDissipation,
        power_mxc: PowerDissipation,
        name: str = "Total fault-tolerant hardware energy",
        short_name: str = "total_ft_energy_kWh",
        description: str = "Total fault-tolerant hardware energy consumption (kWh)",
        unit: str = "kWh",
    ) -> None:
        """
        :param consump_concurrent_cores: no. of concurrent decoding cores required at the consumption stage.
        :param avail_phys_qubits: no. of available physical qubits
        :param num_phys_qubits: no. of required physical qubits
        :param total_ft_time: total fault-tolerant computation time
        :param decoder_tock: decoder tock (sec)
        :param power_4k: power dissipation at the 4K stage
        :param power_mxc: power dissipation at the MXC stage
        """
        super().__init__(name, short_name, description, unit)
        self.consump_concurrent_cores = consump_concurrent_cores
        self.total_ft_time = total_ft_time
        self.power_4k = power_4k
        self.power_mxc = power_mxc

    def get_value(
        self,
        experiment: Experiment,
    ) -> float:
        """Return total fault-tolerant hardware energy consumption over all algorithmic step.

        :param experiment: experiment to calculate the total energy consumption for.
        """
        cooling_efficiency_4kelvin = 500  # W/W cooling for a 4K cryo-cooler
        cooling_efficiency_mxc = 1e9  # W/W cooling for a dilution refrigerator at 0.02K
        # We assume a fixed number of concurrent decoding cores equal to `consump_concurrent_cores` are assigned to all
        # decoding opertions, including graph preparation, consumption, handover, and T-distillation, which operate at a
        # constant TPD of `ref_decode_watt` during `total_ft_time`.
        max_decoder_cores = self.consump_concurrent_cores.get_value(experiment)
        ref_decode_watt = 100
        total_ft_time = self.total_ft_time.get_value(experiment)
        decoder_energy = max_decoder_cores * ref_decode_watt * total_ft_time
        cooling_4k_energy = (
            self.power_4k.get_value(experiment) * total_ft_time * cooling_efficiency_4kelvin * 1000
        )  # already in kW, convert to W
        cooling_mxc_energy = (
            self.power_mxc.get_value(experiment) * total_ft_time * cooling_efficiency_mxc * 1000
        )  # already in kW, convert to W

        # Note: Powers are not essentially the wall power, therefore, we need to consider efficiency
        return (decoder_energy + cooling_4k_energy + cooling_mxc_energy) / 1000 / 3600


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
            "est_time_sec": self.experiment.est_time_sec,
            "version": self.experiment.version,
            "decomp_method": self.experiment.decomp_method,
            "target_p_algo": self.experiment.target_p_algo,
            "dist_widget_name": self.experiment.widget.name,
            "num_pipes_per_intermodule_connection": self.experiment.system_architecture.num_pipes_per_intermodule_connection,  # noqa
            "num_algorithm_steps": self.experiment.num_algorithm_steps,
        }

        if key == "short":
            for resource in self.resources:
                logger.debug(
                    f"The following resource was added: {resource.short_name}: {resource.get_value(self.experiment)}"
                )
            return {
                **experiment_attrs,
                **{resource.short_name: resource.get_value(self.experiment) for resource in self.resources},
            }
        if key == "name":
            return {resource.name: resource.get_value(self.experiment) for resource in self.resources}
        if key == "description":
            return {resource.description: resource.get_value(self.experiment) for resource in self.resources}

    def to_console(self, key="description"):
        """Print results to console."""
        print(
            "\nRRE: Estimated fault-tolerant resources required for the two fridge bilinear quantum and T-transfer "
            "buses architecture with gate-synth at the measurement points are:\n"
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
        resources.append(Distance())
        resources.append(NumLogicalQubits())
        resources.append(NumTFactories())
        resources.append(memory_logical_qubits := MemoryLogicalQubits())
        resources.append(MemoryPhysicalQubits(memory_logical_qubits=memory_logical_qubits))
        resources.append(ancilla_logical_qubits := AncillaLogicalQubits())
        resources.append(AncillaPhysicalQubits(ancilla_logical_qubits=ancilla_logical_qubits))
        resources.append(TBufferLogicalQubits())
        resources.append(TBufferPhysicalQubits())
        resources.append(DistillWidgetLogicalQubits())
        resources.append(DistillWidgetPhysicalQubits())
        resources.append(TotalNumFridges())
        resources.append(NumIntermoduleConnections())
        resources.append(NumPhysicalQubits())
        resources.append(InputLogicalQubits())
        resources.append(DiamondNormEps())
        resources.append(TCount())
        resources.append(TDepth())
        resources.append(InitRZCount())
        resources.append(InitTCount())
        resources.append(InitCliffordCount())
        resources.append(GraphN())
        resources.append(ConsumpScheduleSteps())
        resources.append(PrepScheduleSteps())
        resources.append(NumWidgetSteps())
        resources.append(NumDistinctWidgets())
        resources.append(decoder_tock := DecoderTock())
        resources.append(QuantumIntraTock())
        resources.append(TStateTock())
        resources.append(AvailPhysicalQubits())
        resources.append(AvailLogicalQubits())
        resources.append(unalloc_logical_qubits := UnallocLogicalQubits())
        resources.append(UnallocPhysicalQubits(unalloc_logical_qubits=unalloc_logical_qubits))
        resources.append(alloc_logical_qubits := AllocLogicalQubits())
        resources.append(AllocPhysicalQubits(alloc_logical_qubits=alloc_logical_qubits))
        resources.append(consump_cores := ConsumpStageDecodingCores(decoder_tock=decoder_tock))
        resources.append(ChipArea())
        resources.append(NumberCouplers())
        resources.append(DecodingConsumptionPower(consump_cores=consump_cores))
        resources.append(power_4k := PowerDissipation(thermal_loads_index=0))
        resources.append(power_mxc := PowerDissipation(thermal_loads_index=1))
        resources.append(total_consump_ops_time := TotalConsumpOpsTime())
        resources.append(total_handover_time := TotalHandoverTime())
        resources.append(overall_t_distill_delay := OverallTDistillDelay())
        resources.append(OverallPrepDelay())
        resources.append(
            decode_delay := OverallDecodingDelay(
                decoder_tock=decoder_tock,
                total_consump_ops_time=total_consump_ops_time,
                overall_t_state_delay=overall_t_distill_delay,
            )
        )
        resources.append(
            ft_walltime := FTWallTime(
                total_consump_ops_time=total_consump_ops_time,
                overall_decoding_delay=decode_delay,
                total_handover_time=total_handover_time,
            )
        )
        resources.append(total_ft_time := TotalFTAlgorithmTime(ft_time=ft_walltime))
        resources.append(
            TotalFTEnergy(
                consump_concurrent_cores=consump_cores,
                total_ft_time=total_ft_time,
                power_4k=power_4k,
                power_mxc=power_mxc,
            )
        )
        return resources
