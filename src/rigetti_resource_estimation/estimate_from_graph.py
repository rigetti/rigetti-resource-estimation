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
**Module** ``rigetti_resource_estimation.estimate_from_graph``

Set of classes and methods to perform resource estimations for superconducting hardware given JabalizerSchedGraphItems
and other architectural configurations.

Parts of the following code are inspired and/or customized from the original templates in the open-source references of:
[1] https://latticesurgery.com
[2] https://github.com/alexandrupaler/opensurgery
"""

import re
import math
import logging
from typing import Tuple, Literal, List, Optional, Dict
from dataclasses import dataclass, replace

from scipy.special import lambertw

from rigetti_resource_estimation.graph_utils import JabalizerSchedGraphItems, calculate_t_length_unit  # noqa
from rigetti_resource_estimation import hardware_components as hc
from rigetti_resource_estimation import __version__, Configuration

logger = logging.getLogger(__name__)


@dataclass
class WidgetLoopingResults:
    """A dataclass for keeping track of outputs in `dist_widget_looping`."""

    widget: hc.DistWidget
    d: int
    diamond_norm_eps: float
    graph_items: JabalizerSchedGraphItems
    p_algo: float
    num_steps: int


def gatesynth_const_selector(unitary_decomp_method: str) -> Tuple[float, float]:
    """Set the constants required for gate-synth arbitrary-angle unitary decompositions."""
    if unitary_decomp_method == "gridsynth":
        return (4.0, 0)  # GridSynth method considers the original, "worst-case", approach for arbitrary-angle decomps
    elif unitary_decomp_method == "mixed_fallback":
        return (0.57, 8.83)
    else:
        raise ValueError(f"Unrecognized '1q_unitary_decomposition' method: {unitary_decomp_method}")


def d_from_rates(
    graph_info: JabalizerSchedGraphItems,
    widget: hc.DistWidget,
    p_algo: float,
    p_gates: float,
    k: float,
    p_th: float,
) -> int:
    r"""Calculate code distance given gate-synth eps and graph attributes by solving transcendental equations.

    We present a brief derivation, following Eq. (16) of the original RRE manuscript and assuming the additional
    approximation of widget.cycles>>d at all times.

    The scaling from physical error rate, :math:`p`, to single surface code cycle error rate, :math:`p_{\rm cycle}` is
    governed by the power law given below.

    .. math::

        p_{\rm cycle} = k \left ( \frac{p}{p_th} \right )^{(d + 1)/2},

    resulting in a required code distance of,

    .. math::

        d = \rm{ceil} \left \{ \frac{2}{\ln \left ( \frac{p}{p_th} \right )}  \rm{lambert}\left [ \frac{J \, \ln
        \left (\frac{p}{p_th} \right )}{2 \beta \, k \left( \frac{p}{p_th} \right)^{0.5}} \right] \right \},

    where we denote

    .. math::

        J=\ln \left [\frac{1}{1-p_{\rm algo}} \right],
        \beta = 2 * \delta * C * N,

    and :math:`C` cycles are needed to prepare the distilled T-state, :math:`N` graph nodes need to be
    measured, and :math:`\delta` is the maximum node degree of the graph state. :math:`\rm{lambert}[z]` is the
    real part of the LambertW function for the branch :math:`-1/e < z < 0.`

    :param graph_info: a JabalizerSchedGraphItems object holding the graph and scheduling properties.
    :param widget: selected T-distillation widget.
    :param p_algo: required failure probability for the algorithm.
    :param p_gates: physical error rate for all quantum ops.
    :param k: k coefficient in power law relationship between surface code and physical error rates.
    :param p_th: p_th coefficient, or physical rate threshold, in power law relationship between surface code and
        physical error rates.

    :returns: code distance, d.
    """
    delta = graph_info.delta or 0
    big_n_length = graph_info.t_count() or 0
    logging.debug(
        f"Calculating code distance based on delta={delta}, cycles={widget.cycles}, and big_n_length={big_n_length}."
    )

    beta = 2 * delta * widget.cycles * big_n_length
    big_j = -1 * math.log(1 - p_algo)

    factor1 = 0.5 * math.log(p_gates / p_th)
    # factor2 = 1.0
    factor3 = big_j / (beta * k * math.pow(p_gates / p_th, 0.5))

    lambert_part = lambertw(factor1 * factor3, -1)
    d = math.ceil(1 / factor1 * lambert_part.real)
    if d <= 2:
        raise RuntimeError(f"distance={d} was smaller or equal to 2. No QEC is required!")
    return d


def dist_widget_looping(
    dist_widget_list: List[hc.DistWidget],
    graph_info: JabalizerSchedGraphItems,
    const: Tuple[float, float],
    config: Configuration,
    circuit_fname: str,
) -> WidgetLoopingResults:
    """Loop through distillation widget lookup table to find the right values for `eps` and `d`.

    :param dist_widget_list: A list of DistWidget objects, a.k.a. the T-distillation widget lookup table.
    :param graph_info: A JabalizerSchedGraphItems object containing the complete graph state and scheduling attributes.
    :param const: Set the constants required for the arbitrary-angle unitary decompositions (gate-synth).
    :param config: A Configuration object containing algorithmic and HW requirements. See `__init__.py` for details.
    :param circuit_fname: The stem for the logical circuit file path.

    :returns: a WidgetLoopingResults object, which contains the results of looping through distillation widgets and eps
        values.
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
                graph_info=graph_info, widget=widget, p_algo=p_algo, p_gates=config.p_gates, k=k, p_th=p_th
            )
            # Below, we implement an approximation to the original manuscript equation of
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
                "The followings params were successfully matched:\n"
                f"widget_name: {widget.name}, d={d}, eps={eps}, and t_length_unit={graph_info.t_length_unit}"
            )
            widget_looping_results = WidgetLoopingResults(
                widget=widget,
                d=d,
                diamond_norm_eps=eps,
                graph_items=graph_info,
                p_algo=p_algo,
                num_steps=int(steps),
            )
            return widget_looping_results

    raise ValueError("None of the distillation widgets provided were a match!")


class Experiment:
    """Class to hold collective results for the estimation experiment."""

    def __init__(
        self,
        input_logical_qubits: int,
        graph_info: JabalizerSchedGraphItems,
        circuit_fname: str,
        est_method: str,
        config: Configuration,
        graph_state_opt: Literal["no_compile", "save", "resume"],
        circuit_info: Optional[Dict[str, str]] = None,
    ) -> None:
        r"""
        :param input_logical_qubits: Number of logical qubits in the input circuit.
        :param graph_info: A JabalizerSchedGraphItems object containing the complete graph state and scheduling
            attributes.
        :param circuit_fname: The stem for the logical circuit file path.
        :param est_method: Choice of method to perform the resource estimations. It can be set to either:
            't_counting': Assumes that the graph state attributes of N and Delta are simply equal to the total T-count
            of logical algortihm.
            'jabalizer': Use Jabilizer compilation tools for calculating the graph parameters explicitly.
        :param config: A Configuration object containing algorithmic and HW requirements. See `__init__.py` for details.
        :param graph_state_opt: What to do concerning the graph state compilation. Options are explained below.
            'no_compile': The graph compilation pipeline will not be executed (relevant, e.g., for 't_counting'
            approaches).
            'save': RRE compiles the circuit and attempts to generate the graph state and Pauli Frames info using
            the Jabalizer compiler. The program will save all the graph and scheduling info in a unified JSON file in
            the subdirectory `output/<circuit_fname>/`. The outputs will be named
            `<circuit_fname>_all0init_jabalize.json`.
            'resume': RRE will try to resume the calculations assuming a `<circuit_fname>_all0init_jabalize.json` file
            already exist in `output/<circuit_fname>/` subdirectory.
        :param circuit_info: information from the circuit decomposition process, if any.
        """
        widget_table = hc.build_dist_widgets(config.widget_params)
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
        self.input_log_qubits = input_logical_qubits
        self.graph_items = widget_looping_results.graph_items
        self.diamond_norm_eps = widget_looping_results.diamond_norm_eps
        self.target_p_algo = widget_looping_results.p_algo
        self.num_steps = widget_looping_results.num_steps

        self.phys_gate_error_rate = config.p_gates
        self.decoder_char_timescale_sec = config.tensordecoder_char_timescale_sec

        system_architecture = hc.build_system(widget_looping_results.widget, config.ftqc_params)
        self.system_architecture = system_architecture
        self.consump_sched_times = system_architecture.calc_consump_sched_time(
            graph_items=widget_looping_results.graph_items,
            dist_widget=widget_looping_results.widget,
            distance=widget_looping_results.d,
            num_intermodule_pipes=system_architecture.num_intermodule_pipes,
        )
        self.num_t_factories = system_architecture.calc_num_t_factories(
            widget_q=widget_looping_results.widget.qubits,
            d=widget_looping_results.d,
            req_logical_qubits=int(widget_looping_results.graph_items.delta or 0),
        )
        self.available_logical_qubits = 2 * system_architecture.allocate_logicals_to_modules(
            distance=widget_looping_results.d,
            req_logical_qubits=(widget_looping_results.graph_items.delta or 0),
        )

        self.sims_time_sec = 0.0
        self.decomp_method = config.decomp_method
        self.target_p_algo = config.target_p_algo
        self.graph_state_opt = graph_state_opt

        # In [Bravyi et al., 2014, arXiv:1405.4883], it was shown that bond_dim=8 is enough to saturate best threshold
        # performance for d<=25. In [Chubb 2021, arXiv:2101.04125], bond_dim=48 was used to reach near-ideal threshold
        # for somewhat larger range of distances for square surface code and worst-case noise cases. While bond_dim of
        # O(1) should be enough to saturate the threshold for most cases, [Bravyi et al., 2014, arXiv:1405.4883] has
        # also suggested bond_dim needs to (weakly) grow with d, due to correspondence with Ising model, to reach best
        # performance. Based on the above and some heuristics we make the following choices.
        self.bond_dim = math.floor(0.1 * math.pow(widget_looping_results.d, 0.5)) + 8
        self.circuit_info = circuit_info or dict()

        logger.info("All required attributes for the `experiment` object were assigned.")
