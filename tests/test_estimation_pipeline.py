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

"""Unit tests for the `estimation_pipeline` module of rigetti-resource-estimation."""

import pandas as pd
import pytest
import cirq.testing as ct
import cirq
from unittest.mock import patch

from rigetti_resource_estimation import Configuration, load_yaml_file
from rigetti_resource_estimation.estimation_pipeline import estimation_pipeline
from rigetti_resource_estimation.resources import (
    DefaultResourceCollection,
    ResourceEstimator,
)
from rigetti_resource_estimation.estimate_from_graph import Experiment
from rigetti_resource_estimation.graph_utils import CompilerSchedGraphItems
from rigetti_resource_estimation import translators, transpile

default_transpiler = transpile.CirqTranspiler(translators.DEFAULT_TRANSLATORS)


@pytest.fixture
def get_widgetized_random_circuit():
    circuit = ct.random_circuit(qubits=20, n_moments=100, op_density=0.7)
    # widget_result = widgetization.WidgetizationResult(["W-0"], {"W-0": circuit}, set(circuit.all_qubits()))
    widget_dict = {
        "stitches": {("W-0", "W-0"): 3},
        "widgets": {"W-0": (circuit, 4)},
        "first_widget": "W-0",
        "compiler_tag_table": {},
        "circuit_name": "test_circuit",
        "input_qubits": 20,
    }
    return widget_dict


@pytest.mark.parametrize("log", [None, "INFO", "DEBUG", "CRITICAL", "ERROR"])
@pytest.mark.parametrize("unitary_decomp_method", ["gridsynth", "mixed_fallback"])
def test_estimation_pipeline_tcounting(
    tmp_path,
    log,
    unitary_decomp_method,
):
    """Pytests for frontend estimation_pipeline() function with 't_counting' estimation methodology only."""
    output_csv = tmp_path / "test.csv"
    PARAMS = load_yaml_file()
    params_dict = PARAMS
    # Because our T-counting approach would set num_logical_qubits to the T-count, we need to specify a
    # larger-than-default fridge size for the logical nodes to fit.
    params_dict["ftqc"]["processor"]["num_qubits"] = 3e7
    params_dict["decoder_char_timescale_sec"] = 0.05  # in seconds
    params_dict["1q_unitary_decomp_method"] = unitary_decomp_method
    config = Configuration(params_dict)

    estimation_pipeline(
        log=log,
        output_csv=output_csv,
        est_method="t_counting",
        graph_state_opt="no_compile",
        config=config,
    )

    if output_csv is not None:
        df = pd.read_csv(output_csv, header=0, index_col=None)
        results = df.to_dict()
        assert results["input_logical_qubits"][0] == 4
        assert results["total_avail_physical_qubits"][0] == 6.0e7


def test_estimation_pipeline_cabalizer(tmp_path):
    """Pytests for frontend estimation_pipeline() function with 'cabalizer' estimation method and default circuits."""
    output_csv = tmp_path / "test2.csv"

    PARAMS = load_yaml_file()
    params_dict = PARAMS
    params_dict["decoder_char_timescale_sec"] = 0.05  # in seconds
    config = Configuration(params_dict)

    estimation_pipeline(
        log="DEBUG",
        output_csv=output_csv,
        est_method="cabaliser",
        graph_state_opt="save",
        config=config,
    )

    df = pd.read_csv(output_csv, header=0, index_col=None)  # type: ignore
    results = df.to_dict()

    assert results["rz_count"][0] == 48
    assert results["t_count"][0] == 1132
    assert results["input_logical_qubits"][0] == 4
    assert results["N"][0] == 216
    assert results["num_logical_qubits_per_busrail"][0] == 22
    assert results["total_avail_physical_qubits"][0] == 2000000
    assert results["avail_logical_qubits_per_module"][0] == 3364
    assert results["distance"][0] == 12
    assert results["consumption_schedule_size"][0] == 172
    assert results["preparation_schedule_size"][0] == 172
    assert results["num_t_factories_per_module"][0] == 126


def test_estimator_types(get_widgetized_random_circuit):
    """Pytests for the execution and typing of the estimator function."""
    # Below, we leave other options to DEFAULT values as per params.yaml
    circuit_fname = "test"

    node_items = CompilerSchedGraphItems(
        t_count_init=1000,
        rz_count=100,
        clifford_count_init=1000,
        big_n=12,
        delta=4,
        t_length_unit=50,
    )

    PARAMS = load_yaml_file()
    params_dict = PARAMS
    params_dict["decoder_char_timescale_sec"] = 0.05  # in seconds
    params_dict["1q_unitary_decomp_method"] = "gridsynth"
    config = Configuration(params_dict)
    resources = DefaultResourceCollection().build()
    experiment = Experiment(
        transpiled_widgets=get_widgetized_random_circuit,
        graph_info=node_items,
        circuit_fname=circuit_fname,
        est_method="t_counting",
        config=config,
        graph_state_opt="save",
    )
    estimator = ResourceEstimator(circuit_fname, resources, experiment)
    results = estimator.to_dict("short")

    assert isinstance(results["distance"], int)
    assert isinstance(results["diamond_norm_eps"], float)
    assert isinstance(results["t_count"], int)
    assert isinstance(results["total_allocated_physical_qubits"], int)
    assert isinstance(results["total_allocated_logical_qubits"], int)
    assert isinstance(results["N"], int)
    assert isinstance(results["num_logical_qubits_per_busrail"], int)
    assert isinstance(results["t_distillery_logical_qubits_per_module"], int)
    assert isinstance(results["t_distillery_physical_qubits_per_module"], int)
    assert isinstance(results["num_alloc_physical_qubits_per_module"], int)
    assert isinstance(results["total_ft_energy_kWh"], float)
    assert isinstance(results["num_consump_concurrent_cores_decoding"], int)
    assert isinstance(results["total_consump_ops_sec"], float)
    assert isinstance(results["total_ft_algorithm_time_sec"], float)
    assert isinstance(results["algorithm_step_ft_time_sec"], float)


def test_decompose_and_qbmap_uses_default_qubit_manager_in_estimation_pipeline(get_widgetized_random_circuit, tmp_path):
    """Context used by decompose_once should employ the same cirq.SimpleQubitManager for each decomposition."""
    with patch("cirq.protocols.decompose_protocol.decompose_once", wraps=cirq.decompose_once) as mock_decomp:
        estimation_pipeline(
            widgetized_circ_dict=get_widgetized_random_circuit,
            transpiler=default_transpiler,
            output_csv=tmp_path / "testing.csv",
            graph_state_opt="save",
            est_method="cabaliser",
            params_path="src/rigetti_resource_estimation/params.yaml",
            decomp_context=None,
        )
    qms = [args[1]["context"].qubit_manager for args in mock_decomp.call_args_list]
    assert all([qm == qms[0] for qm in qms])
    assert all([isinstance(qm, cirq.SimpleQubitManager) for qm in qms])


def test_decomposeand_qbmap_uses_passed_qubit_manager_in_estimation_pipeline(get_widgetized_random_circuit, tmp_path):
    """Context used by decompose_once should employ the same qubit manager for all decompositions."""
    qb_manager = cirq.SimpleQubitManager("simple_qm")
    context = cirq.DecompositionContext(qubit_manager=qb_manager)
    with patch("cirq.protocols.decompose_protocol.decompose_once", wraps=cirq.decompose_once) as mock_decomp:
        estimation_pipeline(
            widgetized_circ_dict=get_widgetized_random_circuit,
            transpiler=default_transpiler,
            output_csv=tmp_path / "testing.csv",
            graph_state_opt="save",
            est_method="cabaliser",
            params_path="src/rigetti_resource_estimation/params.yaml",
            decomp_context=context,
        )
    qms = [args[1]["context"].qubit_manager for args in mock_decomp.call_args_list]
    assert all([qm == qb_manager for qm in qms])
