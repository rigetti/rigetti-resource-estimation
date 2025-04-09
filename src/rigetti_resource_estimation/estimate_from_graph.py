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
**Module** ``rigetti_resource_estimation.estimate_from_graph``

Set of classes and methods to perform resource estimations for superconducting hardware given CompilerSchedGraphItems
and other architectural configurations.

Parts of the following code are inspired and/or customized from the original templates in the open-source references of:
[1] https://latticesurgery.com
[2] https://github.com/alexandrupaler/opensurgery
"""

import re
import math
import logging
from typing import Tuple, Literal, List
from dataclasses import dataclass, replace

from rigetti_resource_estimation.graph_utils import CompilerSchedGraphItems, calculate_t_length_unit
from rigetti_resource_estimation import hardware_components as hc
from rigetti_resource_estimation import __version__, Configuration

logger = logging.getLogger(__name__)


@dataclass
class DistWidgetLoopingResults:
    """
    A dataclass for keeping track of outputs in `dist_widget_looping`.

    :param widget: Selected T-distillation widget.
    :param d: Code distance, d.
    :param graph_items: A CompilerSchedGraphItems object holding the graph and scheduling properties.
    :param num_algorithm_steps: Number of repetitions of the original algorithm identified by the user in the filename
        as `_numstep*_`. This will become the number of algorithm repetitions: RRE assumes the user intends to repeat
        the algorithm this many times.
    """

    widget: hc.DistWidget
    d: int
    diamond_norm_eps: float
    graph_items: CompilerSchedGraphItems
    p_algo: float
    num_algorithm_steps: int


def gatesynth_const_selector(unitary_decomp_method: str) -> Tuple[float, float]:
    """Set the constants required for gate synthesis of arbitrary-angle unitary decompositions."""
    if unitary_decomp_method == "gridsynth":
        return (4, 0)  # GridSynth method considers the original, "worst-case", approach for arbitrary-angle decomps
    elif unitary_decomp_method == "mixed_fallback":
        return (0.57, 8.83)
    else:
        raise ValueError(f"Unrecognized '1q_unitary_decomposition' method: {unitary_decomp_method}")


def d_condition(
    graph_info: CompilerSchedGraphItems,
    dist_widget: hc.DistWidget,
    d: int,
    config: Configuration,
) -> bool:
    """Check if error rate condition, inequality (15), RRE paper arXiv:2406.06015, is true for (graph_info,widget,d)."""
    p_algo = config.target_p_algo
    delta = graph_info.delta or 0
    k = config.error_scaling_coeffs["k"]
    p_th = config.error_scaling_coeffs["p_th"]
    system_architecture = hc.build_system(dist_widget, config.ftqc_params)
    num_modules_per_leg = system_architecture.allocate_logicals_to_modules(
        dist_widget=dist_widget,
        d=d,
        delta=delta,
    )  # check the routing of intra-fridge components and get no. of fridges
    intra_components_counts = system_architecture.calc_num_t_factories(
        dist_widget=dist_widget,
        d=d,
        delta=math.ceil(delta / num_modules_per_leg),
    )
    if intra_components_counts.num_t_factories == 0:
        raise RuntimeError(
            "RRE has received a null no. of T-factories per module, making architectural assumptions invalid.\n"
        )
    n_row_qbus = max(math.floor((math.ceil(delta / num_modules_per_leg) + 1) / intra_components_counts.l_qbus), 1)
    n2 = min(
        (
            intra_components_counts.num_t_factories * 4
            if dist_widget.cond_20to4
            else intra_components_counts.num_t_factories
        ),
        n_row_qbus,
    )
    total_t_length = graph_info.t_count() or 0
    j1 = math.log(1 - p_algo)

    logging.debug(
        "Checking the validity of d_cond for the following data:\n"
        f"d={d}, delta={delta}, n2={n2}, total_t_length={total_t_length}, and j1={j1}.\n"
    )

    # Calculating the total number of sequential distillation ops required
    total_seqs_distill = math.ceil(total_t_length / n2)

    # Calculating the total number of sequential graph consumption ops
    total_seqs_consump = graph_info.t_length_unit * math.ceil(graph_info.rz_count / n2) + math.ceil(
        graph_info.t_count_init / n2
    )

    lhs_check = (
        k
        * math.pow(config.p_gates / p_th, (d + 1) / 2)
        * d
        * (
            (graph_info.big_s_prep or 0) * d * 2 * delta
            + (2 * delta + num_modules_per_leg * intra_components_counts.len_transfer_bus)
            * (total_seqs_distill * dist_widget.cycles + total_seqs_consump * d)
        )
        + j1
    )

    return True if lhs_check < 0 else False


def d_from_rates(
    graph_info: CompilerSchedGraphItems,
    widget: hc.DistWidget,
    config: Configuration,
) -> int:
    r"""Calculate the code distance by iterating through gate-synth eps, graph attributes, and T widgets.

    Below, we iteratively find the optimal (smallest) d that satisfies the logarithmic space-time-volume Eq. (15) of
    the RRE manuscript arXiv:2406.06015. We note that Eq. (15) assumes an intra-module architecture that allocates a
    bilinear quantum bus, multiple T-factories for parallel T-state consumption, and a T-transfer bus queueing T
    states. We assume simultaneous T-states could be moved around with O(1) ops and consumed as per the consump
    schedule steps.  For every proposed d, some of the architectural assumptions, i.e. error rate conditions and
    validating the preparation and consumption of T-states for time steps, are checked in this method and some
    separately in `hardware_components.py`.

    :param graph_info: a CompilerSchedGraphItems object holding the graph and scheduling properties.
    :param widget: selected T-distillation widget.
    :param config: A Configuration object containing algorithmic and HW requirements. See `__init__.py` for details.

    :returns: code distance, d.
    """
    delta = graph_info.delta or 0
    total_t_length = graph_info.t_count() or 0
    logging.debug(
        "\nFinding the optimal code distance based on:\n"
        f"delta={delta}, widget={widget.name}, and total_t_length={total_t_length}.\n"
    )

    # Initial guess for d:
    d = 3
    d_condition_check = d_condition(graph_info, widget, d, config)

    while not d_condition_check:
        d = d + 1
        d_condition_check = d_condition(graph_info, widget, d, config)

    if d <= 2:
        raise RuntimeError(f"distance={d} was smaller or equal to 2. No QEC is required!")

    return d


def dist_widget_looping(
    dist_widget_list: List[hc.DistWidget],
    graph_info: CompilerSchedGraphItems,
    const: Tuple[float, float],
    config: Configuration,
    circuit_fname: str,
) -> DistWidgetLoopingResults:
    """Loop through distillation widget lookup table to find the right values for `eps` and `d`.

    :param dist_widget_list: A list of DistWidget objects, a.k.a. the T-distillation widget lookup table.
    :param graph_info: A CompilerSchedGraphItems object containing the complete graph state and scheduling attributes.
    :param const: Set the constants required for the arbitrary-angle unitary decompositions (gate-synth).
    :param config: A Configuration object containing algorithmic and HW requirements. See `__init__.py` for details.
    :param circuit_fname: The stem for the logical circuit file path.

    :returns: a DistWidgetLoopingResults object, which contains the results of looping through distillation widgets and
        eps values.
    """
    logger.info("RRE will now try to find appropriate {widget, eps, d} specs based on the constructed graph state ...")

    d = 0
    p_algo = config.target_p_algo
    k = config.error_scaling_coeffs["k"]
    p_th = config.error_scaling_coeffs["p_th"]

    # Re-assigning p_algo for algorithms with many steps:
    if config.n_steps == "from_filename":
        match = re.search(r"_numstep(\d+)_", circuit_fname)
        steps = match.group(1) if match else ""
    elif isinstance(config.n_steps, int) and config.n_steps > 0:
        steps = config.n_steps
    else:
        raise ValueError("In parameters file, 'n_steps' must be set to either 'from_filename' or positive integers.")
    p_algo = 1 - math.pow(1 - p_algo, 1 / int(steps))

    for widget in dist_widget_list:
        logger.info(f"Trying to match widget:{widget.name} with p_out:{widget.p_out} ...")
        eps = 0.1 if config.fixed_eps is None else config.fixed_eps  # initial guess for the diamond-norm precision, eps
        graph_info = replace(graph_info, t_length_unit=calculate_t_length_unit(const=const, epsilon=eps))
        p_cell = 0
        big_m_cond = False
        while big_m_cond is False:
            d = d_from_rates(
                graph_info=graph_info,
                widget=widget,
                config=config,
            )
            # Below, we implement an approximation to the original RRE manuscript arXiv:2406.06015 equation of
            # p_cell = 1 - math.pow(1 - k * math.pow(config.p_gates / p_th, 0.5 * d + 0.5), d).
            p_cell = d * k * math.pow(config.p_gates / p_th, 0.5 * d + 0.5)
            big_m_cond = p_cell > eps
            if big_m_cond is True:
                logger.info(f"converged for eps={eps}.")
            elif eps < 1e-32 or config.fixed_eps is not None:
                logger.warning(
                    "`eps` became too small and convergence was not achieved for:\n"
                    f"d:{d}, p_cell:{p_cell}, widget:{widget.name}"
                )
                big_m_cond = True
            else:
                eps = eps / 10
                graph_info = replace(graph_info, t_length_unit=calculate_t_length_unit(const=const, epsilon=eps))
        if p_cell > widget.p_out:
            logger.info(
                "The followings params were successfully matched given the inter-module architecture:\n"
                f"widget_name: {widget.name}, d={d}, eps={eps}, and t_length_unit={graph_info.t_length_unit}"
            )
            return DistWidgetLoopingResults(
                widget=widget,
                d=d,
                diamond_norm_eps=eps,
                graph_items=graph_info,
                p_algo=p_algo,
                num_algorithm_steps=int(steps),
            )

    raise ValueError("None of the distillation widgets provided were a match!")


class Experiment:
    """Class to hold collective results for the estimation experiment."""

    def __init__(
        self,
        graph_info: CompilerSchedGraphItems,
        circuit_fname: str,
        est_method: str,
        config: Configuration,
        graph_state_opt: Literal["no_compile", "save", "resume"],
        transpiled_widgets: dict,
    ) -> None:
        r"""
        :param graph_info: A CompilerSchedGraphItems object containing the complete graph state and scheduling
            attributes.
        :param circuit_fname: The stem for the logical circuit file path.
        :param est_method: Choice of method to perform the resource estimations. It can be set to either:
            't_counting': Assumes that the graph state parameters of N (graph size) and Delta (max memory) are equal to
            the total T-count of the logical algorithm.
            'cabalizer': Use Cabalizer FT-compilation tools to calculate the graph properties explicitly.
        :param config: A Configuration object containing algorithmic and HW requirements. See `__init__.py` for details.
        :param graph_state_opt: What to do concerning the graph state compilation. Options are explained below.
            'no_compile': The graph compilation pipeline will not be executed (relevant, e.g., for 't_counting'
            approaches).
            'save': RRE compiles the subcircuit and attempt to generate the graph states and Pauli Frames info through
            Cabaliser compiler. The program will save all outputs in JSON file in the subdirectory
            `output/<circuit_fname>/`. The outputs will be named `<circuit_fname>_all0init_cabalizeframes.json`.
            'resume': RRE will try to resume the calculations assuming an `<circuit_fname>_all0init_cabalizeframes.json`
            already exist in `output/<circuit_fname>/` subdirectory.
        :param transpiled_widgets: RRE standard dict containing transpiled widgets info.
        """
        widget_table = hc.build_dist_widgets(config.widget_params)  # type: ignore
        const = gatesynth_const_selector(config.decomp_method)
        widget_looping_results = dist_widget_looping(
            dist_widget_list=widget_table.widgets,
            graph_info=graph_info,
            const=const,
            config=config,
            circuit_fname=circuit_fname,
        )

        self.version = __version__
        self.circuit_fname = circuit_fname
        self.est_method = est_method

        self.widget = widget_looping_results.widget
        self.distance = widget_looping_results.d
        self.input_logical_qubits = transpiled_widgets["input_qubits"]
        self.graph_items = widget_looping_results.graph_items
        self.diamond_norm_eps = widget_looping_results.diamond_norm_eps
        self.target_p_algo = widget_looping_results.p_algo
        self.num_algorithm_steps = widget_looping_results.num_algorithm_steps

        self.phys_gate_error_rate = config.p_gates
        self.decoder_char_timescale_sec = float(config.decoder_char_timescale_sec)

        system_architecture = hc.build_system(widget_looping_results.widget, config.ftqc_params)
        self.system_architecture = system_architecture
        self.consump_sched_times = system_architecture.calc_consump_sched_times(
            graph_items=widget_looping_results.graph_items,
            dist_widget=widget_looping_results.widget,
            distance=widget_looping_results.d,
            num_pipes_per_intermodule_connection=system_architecture.num_pipes_per_intermodule_connection,
            transpiled_widgets=transpiled_widgets,
        )
        num_modules_per_leg = system_architecture.allocate_logicals_to_modules(
            dist_widget=widget_looping_results.widget,
            d=widget_looping_results.d,
            delta=widget_looping_results.graph_items.delta or 0,
        )
        self.intra_component_counts = system_architecture.calc_num_t_factories(
            dist_widget=widget_looping_results.widget,
            d=widget_looping_results.d,
            delta=math.ceil((widget_looping_results.graph_items.delta or 0) / num_modules_per_leg),
        )
        self.num_modules_per_leg = num_modules_per_leg

        self.est_time_sec = 0.0
        self.decomp_method = config.decomp_method
        self.target_p_algo = config.target_p_algo
        self.graph_state_opt = graph_state_opt

        # In [Bravyi et al., 2014, arXiv:1405.4883], it was shown that bond_dim=8 is enough to saturate best threshold
        # performance for d<=25. In [Chubb 2021, arXiv:2101.04125], bond_dim=48 was used to reach near-ideal threshold
        # for somewhat larger range of distances for square surface code and worst-case noise cases. While bond_dim of
        # O(1) should be enough to saturate the threshold for most cases, [Bravyi et al., 2014, arXiv:1405.4883] has
        # also suggested bond_dim needs to (weakly) grow with d, due to correspondence with Ising model, to reach best
        # performance. Based on the above and some heuristics, we have made the following choices.
        self.bond_dim = math.floor(0.1 * math.pow(widget_looping_results.d, 0.5)) + 8

        self.transpiled_widgets = transpiled_widgets

        logger.info("All required attributes for the `experiment` object were assigned.")
