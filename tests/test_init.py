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

"""Unit tests for rigetti_resource_estimation/__init__.py."""

import pytest
import yaml
from rigetti_resource_estimation import Configuration


@pytest.fixture
def example_params():
    """Create an example params dictionary for use in testing."""
    params = {
        "1q_unitary_decomp_method": "A",
        "fixed_eps": 1.0,
        "ftqc": "C",
        "distill_widget_table": "D",
    }
    return params


@pytest.fixture
def example_yaml_path(example_params, tmp_path):  # pylint: disable=W0621
    """Create temporary directory for test files."""
    temporary_dir = tmp_path
    temporary_dir.mkdir(exist_ok=True)
    example_yaml = temporary_dir / "example.yaml"
    with open(example_yaml, "w", encoding="utf8") as yaml_file:
        yaml.safe_dump(example_params, yaml_file)
    return example_yaml


class TestConfiguration:
    """Test the Configuration class."""

    @staticmethod
    def compare_and_assert_equal(config: Configuration, params: dict) -> None:
        """Test the passed Configuration instance against its params.

        This is just a helper method to reduce repeating this type of test multiple times.
        :param config: Configuration instance to test.
        :param params: true values of the parameters to compare against.
        """
        for param, value in params.items():
            if param == "1q_unitary_decomp_method":
                param = "decomp_method"
            if param == "ftqc":
                param = "ftqc_params"
            if param == "distill_widget_table":
                param = "widget_params"
            should_be = value
            actual = getattr(config, param)
            assert actual == should_be

    def test_init_with_dict(self, example_params) -> None:  # pylint: disable=W0621
        """Test Configuration instantiation via passed dictionary."""
        config = Configuration(example_params)
        self.compare_and_assert_equal(config, example_params)
