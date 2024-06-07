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

"""Unit tests for the `estimation_pipeline` module of rigetti-resource-estimation."""

import pandas as pd
import pytest

from rigetti_resource_estimation import Configuration, load_yaml_file
from rigetti_resource_estimation.estimation_pipeline import estimation_pipeline
from rigetti_resource_estimation.resources import (
    DefaultResourceCollection,
    ResourceEstimator,
)
from rigetti_resource_estimation.estimate_from_graph import Experiment
from rigetti_resource_estimation.graph_utils import JabalizerSchedGraphItems

QASM_PATH = "./tests/input/qft10.qasm"


@pytest.mark.parametrize("qasm", [None, QASM_PATH])
@pytest.mark.parametrize("log", [None, "INFO", "DEBUG", "CRITICAL", "ERROR"])
@pytest.mark.parametrize("output_csv", [None, "test.csv"])
@pytest.mark.parametrize("unitary_decomp_method", ["gridsynth", "mixed_fallback"])
def test_estimation_pipeline_tcounting(
    tmp_path,
    qasm,
    log,
    output_csv,
    unitary_decomp_method,
):
    """Pytests for frontend estimation_pipeline() function with 't_counting' estimation methodology only."""
    if output_csv is not None:
        output_csv = tmp_path / "test.csv"

    PARAMS = load_yaml_file()
    params_dict = PARAMS
    # Because our T-counting approach would set req_logical_qubits to the T-count, we need to specify a
    # larger-then-default fridge size for the logical nodes to fit.
    params_dict["ftqc"]["processor"]["num_qubits"] = 3e7
    params_dict["tensordecoder_char_timescale_sec"] = 0.05  # in seconds
    params_dict["1q_unitary_decomp_method"] = unitary_decomp_method
    config = Configuration(params_dict)

    estimation_pipeline(
        circ_path=qasm,
        log=log,
        output_csv=output_csv,
        est_method="t_counting",
        graph_state_opt="no_compile",
        config=config,
    )

    if output_csv is not None:
        df = pd.read_csv(output_csv, header=0, index_col=None, squeeze=True)  # type: ignore
        results = df.to_dict()

        expected_input_logical_q = 10 if qasm else 4

        assert results["input_log_qubits"][0] == expected_input_logical_q
        assert results["avail_physical_qubits"][0] == 6e7


def test_estimation_pipeline_jabalizer(tmp_path):
    """Pytests for frontend estimation_pipeline() function with the 'jabalizer' est method and default circuits."""
    output_csv = tmp_path / "test2.csv"

    PARAMS = load_yaml_file()
    params_dict = PARAMS
    params_dict["tensordecoder_char_timescale_sec"] = 0.05  # in seconds
    config = Configuration(params_dict)

    estimation_pipeline(
        log="DEBUG",
        output_csv=output_csv,
        est_method="jabalizer",
        graph_state_opt="save",
        config=config,
    )

    df = pd.read_csv(output_csv, header=0, index_col=None, squeeze=True)  # type: ignore
    results = df.to_dict()

    assert results["rz_count"][0] == 9
    assert results["t_count"][0] == 198
    assert results["input_log_qubits"][0] == 4
    assert results["N"][0] == 30
    assert results["required_logical_qubits"][0] == 27
    assert results["avail_physical_qubits"][0] == 2000000
    assert results["avail_logical_qubits"][0] == 4290
    assert results["distance"][0] == 10
    assert results["S_consump"][0] == 8
    assert results["num_t_factories"][0] == 366


def test_estimator_types():
    """Pytests for the execution and typing of the estimator function."""
    # Below, we leave other options to DEFAULT values in params.yaml
    circuit_fname = "test"

    node_items = JabalizerSchedGraphItems(
        t_count_init=1000,
        rz_count=100,
        clifford_count_init=1000,
        big_n=12,
        delta=4,
        t_length_unit=50,
    )

    PARAMS = load_yaml_file()
    params_dict = PARAMS
    params_dict["tensordecoder_char_timescale_sec"] = 0.05  # in seconds
    params_dict["1q_unitary_decomp_method"] = "gridsynth"
    config = Configuration(params_dict)
    resources = DefaultResourceCollection().build()
    experiment = Experiment(
        input_logical_qubits=3,
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
    assert isinstance(results["total_req_physical_qubits"], int)
    assert isinstance(results["N"], int)
    assert isinstance(results["required_logical_qubits"], int)
    assert isinstance(results["wq"], int)
    assert isinstance(results["total_req_physical_qubits"], int)
    assert isinstance(results["distill_decoding_maxmem_MB"], float)
    assert isinstance(results["total_ft_energy_kWh"], float)
    assert isinstance(results["distill_concurrentcores_decoding"], int)
    assert isinstance(results["tot_intra_q_ops_sec"], float)
    assert isinstance(results["full_evolution_time_sec"], float)
    assert isinstance(results["total_ft_time_sec"], float)
