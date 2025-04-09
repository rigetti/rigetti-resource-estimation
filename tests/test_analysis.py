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

"""Unit tests for the analysis module of rigetti-resource-estimation."""

from os import path
from pathlib import Path
from unittest.mock import patch, PropertyMock
import shutil
import pytest

import pandas as pd

import rigetti_resource_estimation.analysis as analysis


@pytest.fixture
def controller():
    """Create SweepController instance for testing."""
    param = "level1.level2.level3.param"
    values = "1,2,3,4"
    return analysis.SweepController(
        param=param,
        values=values,
        output_csv="output_directory/output.csv",
    )


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test files."""
    output_dir = tmp_path
    output_dir.mkdir(exist_ok=True)
    shutil.copy2(
        path.join(path.dirname(path.realpath(__file__)), "../src/rigetti_resource_estimation/params.yaml"),
        output_dir,
    )
    return output_dir


class TestSweepController:
    """Test that SweepController is setup and configured properly."""

    def test_param_key_path(self, controller):  # pylint: disable=W0621
        """Ensure the passed parameter is split into a path properly."""
        should_be = ["level1", "level2", "level3", "param"]
        actual = controller.param_key_path

        assert actual == should_be

    def test_param_for_path(self, controller):  # pylint: disable=W0621
        """Ensure the parameter is formatted for use in an output path properly."""
        should_be = "level1-level2-level3-param"
        actual = controller.param_for_path

        assert actual == should_be

    def test_value_path_pairs(self, controller):  # pylint: disable=W0621
        """Ensure pairs of values and paths to be looped over are correct."""
        should_be = [
            (1.0, Path("output_directory/output_level1-level2-level3-param1.csv")),
            (2.0, Path("output_directory/output_level1-level2-level3-param2.csv")),
            (3.0, Path("output_directory/output_level1-level2-level3-param3.csv")),
            (4.0, Path("output_directory/output_level1-level2-level3-param4.csv")),
        ]
        actual = controller.value_path_pairs

        assert actual == should_be

    def test_output_attribute_construction(self, controller):  # pylint: disable=W0621
        """Ensure that SweepController attributes are initialized properly."""
        should_be = [
            Path("output_directory/output.csv"),
            Path("output_directory"),
            "output",
        ]
        actual = [
            controller.output_filepath,
            controller.output_dir,
            controller.file_name,
        ]

        assert actual == should_be


class TestPerformSweep:
    """Test case for the perform_sweep function in analysis.py."""

    parameters = []
    output_dir = ""

    def base_estimate(self) -> pd.DataFrame:
        """Return mocked resource estimate.

        This data is then multiplied by the swept parameter value to simulate how RRE would return different
        estimates from different parameter values.
        """
        estimates = pd.DataFrame({"qubits": [1], "phys_qubits": [10], "tock": [100]})
        return estimates

    def mock_estimate_pipeline(self, *args, **kwargs) -> None:  # pylint: disable=W0613
        """
        Mock for estimate_pipeline called in perform_sweep.

        Mocks the execution and result of the resource estimation pipeline.
        """
        try:
            param = next(self.parameters)  # type: ignore
            df = self.base_estimate() * int(param)
            df.to_csv(self.output_dir / f"result_{param}.csv", index=False)  # type: ignore
        except StopIteration:
            return

    @patch("rigetti_resource_estimation.analysis.estimation_pipeline")
    @patch(
        "rigetti_resource_estimation.analysis.SweepController.PARAMS_DIR",
        new_callable=PropertyMock,
    )
    def test_sweep(self, mock_params_dir, mock_estimation_pipeline, temp_dir):  # pylint: disable=W0621
        """Ensure that sweep performs as expected by mocking out RRE results."""
        # Set up arbitrary parameters for test
        parameters = ["1", "2", "3"]
        self.parameters = iter(parameters)
        parameter = "ftqc.intermodule_tock_sec"  # arbitrary, but must be in params.yaml

        # Set up mocked call to RRE and mocked SweepController params
        mock_estimation_pipeline.side_effect = self.mock_estimate_pipeline
        mock_params_dir.return_value = temp_dir
        self.output_dir = temp_dir

        # call analysis_perform_sweep
        analysis.perform_sweep(
            parameter,
            ",".join(parameters),
            output_csv=str(temp_dir / "output.csv"),
        )
        actual = pd.read_csv(self.output_dir / "combined.csv", index_col=False)

        # Create expected result (concatenate mocked results of RRE)
        should_be = pd.concat([self.base_estimate() * int(param) for param in parameters])

        # sort to ensure concatenation order doesn't cause failure
        actual_sorted = actual.sort_values(by=list(actual.columns)).reset_index(drop=True)
        should_be_sorted = should_be.sort_values(by=list(should_be.columns)).reset_index(drop=True)

        # test
        pd.testing.assert_frame_equal(actual_sorted, should_be_sorted)
