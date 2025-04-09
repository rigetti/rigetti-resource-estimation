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
**Module** ``rigetti_resource_estimation.estimation_pipeline``

Front-end script for fault-tolerant hardware resource estimations featuring functions to perform a full estimation
pipeline.

The module is designed to accept logical-level gate instructions in the form of QASM or JSON circuits. It then
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
import json
from time import perf_counter
from pathlib import Path
from typing import Optional, Literal, Union

import argh
from qiskit import QuantumCircuit

from rigetti_resource_estimation import (
    PARAMS_YAML_FILENAME,
    transpile,
    graph_utils,
    Configuration,
    load_yaml_file,
)
from rigetti_resource_estimation.resources import ResourceEstimator, DefaultResourceCollection
from rigetti_resource_estimation.estimate_from_graph import Experiment

logger = logging.getLogger(__name__)


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
    circ_path: Optional[str] = None,
    output_csv: Optional[str] = None,
    graph_state_opt: Literal["no_compile", "save", "resume"] = "save",
    est_method: Literal["t_counting", "jabalizer"] = "jabalizer",
    params_path: Optional[Union[str, Path]] = None,
    config: Optional[Configuration] = None,
    decomposed_bb: bool = False,
    pcorrections_flag: bool = False,
    log: Optional[str] = None,
) -> None:
    r"""Perform a complete fault-tolerant resource estimation pipeline given a `config` object.

    **This is Rigetti Resource Estimation's (RRE) main entry point.**

    :param circ_path: Filepath for an input logical circuit. Provide the circuit in one of the two formats listed below.
        If no circuit is provided, the default will become the QFT4 algorithm, which contains T-gates, small-angle
        rotations, and Cliffords, providing a good test case.
        Accepted formats are:
        #. OpenQASM2.0: Standard IBM's OpenQASM v2.0 format -- custom gates are not supported.
        #. RRE custom JSON with decomposed QASMs: This JSON file contains subcircuit QASM strings for the serial,
        time-like, decomposition of an algorithm up to a maximum width. We provide an example of such JSON in
        `./examples/input/qft10-decomposed3.json`. You must set the option `decomposed_bb` to
        True telling RRE you have provided a widgetized JSON containing all the building blocks.

    :param output_csv: If provided, RRE will create and write the estimation results to this file using a
        CSV format specified in `resources.py`. RRE will append to the file if it already exists.
    :param graph_state_opt: What to do concerning the graph state compilation. Options are explained below.
        'no_compile': The graph compilation pipeline will not be executed (relevant, e.g., for 't_counting'
        approaches).
        'save': RRE compiles the circuit and attempts to generate the graph state and Pauli Frames info using
        the Jabalizer compiler. The program will save all the graph and scheduling info in a unified JSON file in
        the subdirectory `output/<circuit_fname>/`. The outputs will be named
        `<circuit_fname>_all0init_jabalize.json`.
        'resume': RRE will try to resume the calculations assuming a `<circuit_fname>_all0init_jabalize.json` file
        already exist in `output/<circuit_fname>/` subdirectory.
    :param est_method: Choice of method to perform the resource estimations. It can be set to either:
            't_counting': Assumes that the graph state parameters of N and Delta are equal to the total T-count
            of the logical algorithm.
            'jabalizer': Use Jabalizer compilation tools to calculate the graph parameters explicitly.
    :param params_path: Filepath to the YAML file from which we load the hardware and logical parameters to the `config`
        object. If 'None', parameters will get loaded from a default `src/rigetti_resource_estimation/params.yaml` file
        included with this package.
    :param config: A Configuration object with algorithmic and hardware requirements. See `__init__.py` for details.
    :param decomposed_bb: If set True, RRE will assume the file provided in `circ_path` option is an 'RRE custom JSON
        with decomposed QASMs' with all building block subcircuits and the ordering listed.
    :param pcorrections_flag: Whether or not to track Pauli correction frames during compilation and include them in
        the output Jabalize JSON (applies only to the `jabalizer` compiler and estimation method).
    :param log: The logging level requested. Can be left unset for no logging or to a valid logging level: 'INFO',
        'WARNING', 'DEBUG', 'ERROR', and 'CRITICAL'.
    """
    if log is not None:
        numeric_level = getattr(logging, log, None)
        logging.basicConfig(level=numeric_level)

    filename = PARAMS_YAML_FILENAME if params_path is None else Path(params_path).name

    print(
        f"\nRRE: ESTIMATION STEP0: Loading architectural configs from `{filename}` and reading the input logical"
        " circuit(s) ...\n"
    )

    if config is None:
        config = Configuration(load_yaml_file(params_path))

    # Load the input circuit
    circuits_qasm = []
    circuit_info = {}
    num_subcircuits = 1
    if circ_path is not None:
        circuit_fname = Path(circ_path).stem
        with open(circ_path, "r", encoding="utf8") as file:
            if decomposed_bb:
                circuit_info = json.load(file)
                subcircuits = circuit_info.pop("ordered_subcircuit_qasms")
                num_subcircuits = len(subcircuits)
                for subcircuit in subcircuits:
                    circuits_qasm.append(subcircuit)
                large_info = circuit_info.pop("large_info_keys", [])
                # remove large info from data so it is not added to experiment and reported in estimates
                for info in large_info:
                    circuit_info.pop(info)
            else:
                circuits_qasm.append(file.read())
                circuit_info = dict()
    else:
        # Setting default logical QASM as the QFT4 algorithm -- override by specifying an input circuit.
        circuits_qasm.append(
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        h q[3];
        cp(pi/2) q[3],q[2];
        h q[2];
        cp(pi/4) q[3],q[1];
        cp(pi/2) q[2],q[1];
        h q[1];
        cp(pi/8) q[3],q[0];
        cp(pi/4) q[2],q[0];
        cp(pi/2) q[1],q[0];
        h q[0];
        swap q[0],q[3];
        swap q[1],q[2];
        """
        )
        circuit_fname = "DEFAULT_CIRCUIT"

    # Printing the input circuit for sanity check purposes
    logger.debug(f"The input logical circuit is:\n{circuits_qasm}\n")

    transpiled_circuits_qasm = []
    initial_t_count = 0
    arbitrary_rot_count = 0
    initial_clifford_count = 0

    simulation_start_t = perf_counter()

    print(
        "RRE: ESTIMATION STEP1: Parse the circuit, replace unsupported, and count gates for FT-compiler usage as"
        " needed ...\n"
    )

    for ii, circ_qasm in enumerate(circuits_qasm):
        transpiled_qasm_counts = transpile.precompile_circuit(circ_qasm)
        logger.info(
            f"Exact transpilation for (sub)circuit{ii} resulted in:\n"
            f"Transpiled qasm:\n{transpiled_qasm_counts.transpiled_qasm}\n"
            f"Initial T's: {transpiled_qasm_counts.init_t_count}, "
            f"non-Clifford Rz's: {transpiled_qasm_counts.init_rz_count}, "
            f"Clifford gates: {transpiled_qasm_counts.init_clifford_count}"
        )
        initial_t_count += transpiled_qasm_counts.init_t_count
        arbitrary_rot_count += transpiled_qasm_counts.init_rz_count
        initial_clifford_count += transpiled_qasm_counts.init_clifford_count
        transpiled_circuits_qasm.append(transpiled_qasm_counts.transpiled_qasm)

    print(
        "RRE: ESTIMATION STEP2: FT-compiler performs logical-level computations to evaluate graph state and T-factory"
        " properties ...\n"
    )

    graph_info = graph_utils.find_graph_info(
        est_method=est_method,
        circuits_qasm=transpiled_circuits_qasm,
        graph_state_opt=graph_state_opt,
        t_count_init=initial_t_count,
        clifford_count_init=initial_clifford_count,
        rz_count=arbitrary_rot_count,
        circuit_fname=circuit_fname,
        num_subcircuits=num_subcircuits,
        pcorrections_flag=pcorrections_flag,
    )

    print(
        "RRE: ESTIMATION STEP3: Estimating physical resources required based on the graph, T-factory instructions, and"
        " FTQC system properties ...\n"
    )
    input_log_qubits = QuantumCircuit.from_qasm_str(transpiled_circuits_qasm[0]).num_qubits

    # Parts of the following block are inspired and/or customized from the original templates in latticesurgery.com.
    resources = DefaultResourceCollection().build()
    experiment = Experiment(
        input_logical_qubits=input_log_qubits,
        graph_info=graph_info,
        circuit_fname=circuit_fname,
        est_method=est_method,
        config=config,
        graph_state_opt=graph_state_opt,
        circuit_info=circuit_info,
    )
    estimator = ResourceEstimator(circuit_fname, resources, experiment)
    experiment.sims_time_sec = perf_counter() - simulation_start_t

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
