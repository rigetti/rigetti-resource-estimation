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
**Module** ``rigetti_resource_estimation.estimation_pipeline``

Front-end script for fault-tolerant hardware resource estimations featuring functions to perform a full estimation
pipeline.

The module is designed to accept logical-level gate instructions in a widgetized and cirq-based JSON format. It then
pre-processes the input based on fault-tolerant compiler requirements, replacing unsupported gates, compiles the
circuit to a graph state representation of surface code and MBQC ops [1-3], and finally, estimates the required
resources based on the graph attributes and a user-specified config object. The final output is console output, and
optional CSV containing a variety of space-time and hardware resources, including a number of physical and logical
qubits as well as runtime estimations.

[1] Simon Anders and Hans J. Briegel, Fast Simulation of Stabilizer Circuits Using a Graph-State Representation, Phys.
Rev. A 73, 022334 (2006).
[2] Alexandru Paler et al., Fault-tolerant, high-level quantum circuits: form, compilation and description, Quantum Sci.
Technol. 2, 025003 (2017).
[3] Madhav Krishnan Vijayan et al., Compilation of algorithm-specific graph states for quantum circuits,
arXiv:2209.07345 (2022).
"""

import sys
import logging

from time import perf_counter
from pathlib import Path
from typing import Optional, Literal, Union

import argh
import cirq

from rigetti_resource_estimation import (
    PARAMS_YAML_FILENAME,
    decomposers,
    translators,
    transpile,
    graph_utils,
    Configuration,
    load_yaml_file,
)
from rigetti_resource_estimation.resources import ResourceEstimator, DefaultResourceCollection
from rigetti_resource_estimation.estimate_from_graph import Experiment


logger = logging.getLogger(__name__)
default_transpiler = transpile.CirqTranspiler(translators.DEFAULT_TRANSLATORS)


def print_estimations(
    estimator: ResourceEstimator,
    output_csv: Optional[str] = None,
) -> None:
    """Print resource estimation results to stdout and, optionally, into an CSV output.

    :param estimator: A ResourceEstimator object holding resource estimations results.
    :param output_csv: Set the output CSV file to store estimation results.
    """
    if output_csv is not None:
        mode = "a" if Path(output_csv).exists() else "w"
        estimator.to_csv(output_csv, mode=mode)
    estimator.to_console(key="description")


def estimation_pipeline(
    widgetized_circ_dict: Optional[dict] = None,
    transpiler: transpile.CirqTranspiler = default_transpiler,
    output_csv: Optional[str] = None,
    graph_state_opt: Literal["no_compile", "save", "resume"] = "save",
    est_method: Literal["t_counting", "cabaliser"] = "cabaliser",
    params_path: Optional[Union[str, Path]] = None,
    config: Optional[Configuration] = None,
    log: Optional[str] = None,
    decomp_context: Optional[cirq.DecompositionContext] = None,
) -> None:
    r"""Perform a complete fault-tolerant resource estimation pipeline given a `config` object.

    **This is Rigetti Resource Estimation's (RRE) main entry point.**

    :param widgetized_circ_dict: A custom RRE dict that contains (widgetized) logical circuits and their ordering.
        If no object is provided, the default will become a dict containing QFT4 algorithm, as below, which contains T,
        arbitrary-angle rotations, and Clifford gates providing a good test case in cirq format. Note we have
        intentionally specified individual gates rather than using built-in functions to help providing a flexible test
        case.
        The dict must contain subcircuit cirq objects, i.e. widgets, for a time-like, decomposition of an algorithm up
        to a maximum logical gate count in a field with a key `widgets`, which also contains widget identifiers. We
        provide an example of such dict below as `widgetized_qft4_dict`. You must also provide the ordering of distinct
        widgets through their identifiers in a key that is named `order`. Other entries must exist but can be left with
        any value as RRE's transpile module will refill them.
        Throughout RRE, we assume two levels of time-like decomposition exists:
        #. Algorithm repetitions: This is how many times the user wants the inputted logical algorithm to be repeated
        # (DEFAULT 1). The user can add a pattern of '_numstep(\d+)_' to the filename of `circ_path` to change this
        # parameter. Since the parameter `p_algo` is always the failure probability budget for the complete algorithm,
        # RRE will then automatically re-scale the failure probability per logical circuit so that this many repetitions
        # of the circuit lead to `p_algo`. RRE will also calculate the full evolution runtime using this parameter.
        #. Widget repetitions or time steps: If the circuit is widgetized, this is how many repetitions of the widgets
        #  or how many time steps exist in the inputted algorithm. This parameter is extracted from RRE custom JSON.
    :param transpiler: The circuit transpiler to use.
    :param output_csv: If provided, RRE will create and write the estimation results to this file using a
        CSV format specified in `resources.py`. RRE will append to the file if it already exists.
    :param graph_state_opt: What to do concerning the graph state compilation. Options are explained below.
        'no_compile': The graph compilation pipeline will not be executed (relevant, e.g., for 't_counting'
        approaches).
        'save': RRE compiles the subcircuit and attempt to generate the graph states and Pauli Frames info through
        Cabaliser compiler. The program will save all outputs in JSON file in the subdirectory
        `output/<circuit_fname>/`. The outputs will be named `<circuit_fname>_all0init_cabalizeframes.json`.
        'resume': RRE will try to resume the calculations assuming an `<circuit_fname>_all0init_cabalizeframes.json`
        already exist in `output/<circuit_fname>/` subdirectory.
    :param est_method: Choice of method to perform the resource estimations. It can be set to either:
            't_counting': Assumes that the graph state parameters of N (graph size) and Delta (max memory) are equal to
            the total T-count of the logical algorithm.
            'cabaliser': Use Cabalizer FT-compilation tools to calculate the graph properties explicitly.
    :param params_path: Filepath to the YAML file from which we load the hardware and logical parameters to the `config`
        object. If 'None', parameters will get loaded from a default `src/rigetti_resource_estimation/params.yaml` file
        included with this package.
    :param config: A Configuration object with algorithmic and hardware requirements. See `__init__.py` for details.
    :param log: The logging level requested. Can be left unset for no logging or to a valid logging level: 'INFO',
        'WARNING', 'DEBUG', 'ERROR', and 'CRITICAL'.
    :param decomp_context: decomposition context to handle the qubit management during full decomposition of widgets.
    """
    if log is not None:
        numeric_level = getattr(logging, log, None)
        logging.basicConfig(level=numeric_level)

    filename = PARAMS_YAML_FILENAME if params_path is None else Path(params_path).name
    print(f"\nRRE STEP0: Loading architectural configs from `{filename}` and reading the input logical circuit(s)\n")

    if config is None:
        config = Configuration(load_yaml_file(params_path))

    # Load the input circuit
    circuit_decomposer = decomposers.CirqDecomposer(decomposers.DEFAULT_DECOMPOSERS)

    if widgetized_circ_dict is None:
        # Setting the default input algorithm as a dict with two widgets of size 4 logical qubits (namely W1 and W2).
        # The overall algorithm pattern is "W1W1W1W2W1W1W1W2W1W1W1W1", where W1 is a non-Clifford circuit manually set
        # below and W2 is QFT4. You may override this by editing the circ's and dict values in `widgetized_circ_dict`.
        no_in_qubits = 4
        qq = cirq.LineQubit.range(no_in_qubits)
        default_circ1 = cirq.Circuit(
            cirq.H(qq[2]),
            cirq.CZ(qq[3], qq[1]) ** 0.125,
            cirq.T(qq[0]),
            cirq.SWAP(qq[1], qq[0]),
        )
        default_circ2 = cirq.qft(*qq)
        widgetized_circ_dict = {
            "widgets": {"W1": (default_circ1, 10), "W2": (default_circ2, 2)},
            "stitches": {("W1", "W1"): 7, ("W1", "W2"): 2, ("W2", "W1"): 2},
            "first_widget": "W1",  # Identifying the very first widget required to calculate initial prep time at Step 0
            "circuit_name": "qft4_random_reps",
            "compiler_tag_table": {},
            "input_qubits": no_in_qubits,
            "init_t_count": None,
            "init_rz_count": None,
            "init_clifford_count": None,
        }

    simulation_start_t = perf_counter()

    print(
        "RRE STEP1: Parse the widgetized circ dict, transpile unsupported, and count gates for FT-compiler"
        "  usage as needed\n"
    )

    transpiled_widgets = widgetized_circ_dict
    initial_t_counts = {}
    arbitrary_rot_counts = {}
    initial_clifford_counts = {}
    qbs_appearing = set()
    for label, (circuit, reps) in widgetized_circ_dict["widgets"].items():
        decomposed_circ, qb_map = circuit_decomposer.decompose_and_qb_map(
            circuit=circuit, keep=transpiler.is_compatible, context=decomp_context
        )
        qbs_appearing.update(decomposed_circ.all_qubits())  # type: ignore
        transpiled_cirq_counts = transpiler.transpile(circuit=decomposed_circ, qb_map=qb_map)  # type: ignore
        logger.info(
            f"Exact transpilation for widget:{label} resulted in:\n"
            f"\tTranspiled subcircuit:\n{transpiled_cirq_counts.transpiled_cirq}\n"
            f"\tInitial T count: {transpiled_cirq_counts.init_t_count}\n"
            f"\tnon-Clifford Rz count: {transpiled_cirq_counts.init_rz_count}\n"
            f"\tClifford gates: {transpiled_cirq_counts.init_clifford_count}\n"
            f"\tcompiler_tag_table: {transpiled_cirq_counts.metadata}\n"
        )
        initial_t_counts[label] = transpiled_cirq_counts.init_t_count
        arbitrary_rot_counts[label] = transpiled_cirq_counts.init_rz_count
        initial_clifford_counts[label] = transpiled_cirq_counts.init_clifford_count
        transpiled_widgets["widgets"][label] = (transpiled_cirq_counts.transpiled_cirq, reps)
        # "compiler_tag_table" is fixed among all transpiled widgets and we are only interested in the longest.
        if len(transpiled_cirq_counts.metadata.values()) > len(transpiled_widgets["compiler_tag_table"].values()):  # type: ignore # noqa
            transpiled_widgets["compiler_tag_table"] = transpiled_cirq_counts.metadata

    transpiled_widgets["widget_t_counts"] = initial_t_counts
    transpiled_widgets["widget_z_rot_counts"] = arbitrary_rot_counts
    transpiled_widgets["widget_clifford_counts"] = initial_clifford_counts
    transpiled_widgets["input_qubits"] = len(qbs_appearing)
    transpiled_widgets["init_t_count"] = sum(
        [initial_t_counts[label] * reps for label, (_, reps) in transpiled_widgets["widgets"].items()]
    )
    transpiled_widgets["init_rz_count"] = sum(
        [arbitrary_rot_counts[label] * reps for label, (_, reps) in transpiled_widgets["widgets"].items()]
    )
    transpiled_widgets["init_clifford_count"] = sum(
        [initial_clifford_counts[label] * reps for label, (_, reps) in transpiled_widgets["widgets"].items()]
    )
    print(
        "RRE logical gate counts:\n"
        f"\tT:{transpiled_widgets['init_t_count']},Rz:{transpiled_widgets['init_rz_count']},Clifford:{transpiled_widgets['init_clifford_count']}\n"
    )
    logger.info(f"Exact transpilation for input circuit resulted in `transpiled_widgets`:\n{transpiled_widgets}\n")

    print(
        "RRE STEP2: FT-compiler performs logical-level computations to evaluate graph state and T-factory"
        " properties\n"
    )
    graph_info = graph_utils.find_graph_info(
        est_method=est_method,
        transpiled_widgets=transpiled_widgets,
        graph_state_opt=graph_state_opt,
    )

    print(
        "\nRRE STEP3: Estimating physical resources required based on the graph, T-factory allocations, and"
        " FTQC system properties\n"
    )
    # Parts of the following block are inspired and/or customized from the original templates in latticesurgery.com.
    resources = DefaultResourceCollection().build()
    experiment = Experiment(
        graph_info=graph_info,
        circuit_fname=transpiled_widgets["circuit_name"],
        est_method=est_method,
        config=config,
        graph_state_opt=graph_state_opt,
        transpiled_widgets=transpiled_widgets,
    )
    estimator = ResourceEstimator(transpiled_widgets["circuit_name"], resources, experiment)
    experiment.est_time_sec = perf_counter() - simulation_start_t

    print_estimations(estimator=estimator, output_csv=output_csv)


if __name__ == "__main__":
    try:
        argh.dispatch_command(estimation_pipeline)
        sys.exit(0)
    except AssertionError as exception:
        logger.exception(exception)
        logger.critical("The program encountered a self-consistency issue and was aborted.")
        sys.exit(-1)
    except Exception as exception:  # pylint: disable=W0718
        logger.exception(exception)
        logger.critical("The program encountered a run-time error and was aborted.")
        sys.exit(1)
