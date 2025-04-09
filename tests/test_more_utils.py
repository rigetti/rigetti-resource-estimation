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

"""Unit tests for the `more_utils` module of rigetti-resource-estimation."""


import yaml

import pandas as pd
import pytest

import rigetti_resource_estimation as rre
import rigetti_resource_estimation.more_utils as utils


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test files."""
    output_dir = tmp_path
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture(autouse=True)
def example_csvs(temp_dir):  # pylint: disable=W0621
    """Create example CSVs for testing."""
    output_dir = temp_dir
    data1 = {"num_qubits": [1, 2, 3], "phys_qubits": [4, 5, 6]}
    data2 = {"num_qubits": [7, 8, 9], "phys_qubits": [10, 20, 30]}
    pd.DataFrame(data1).to_csv(output_dir / "file1.csv", index=False)
    pd.DataFrame(data2).to_csv(output_dir / "file2.csv", index=False)


@pytest.fixture
def example_dicts(temp_dir):  # pylint: disable=W0621
    """Create example dictionaries for testing update function."""
    output_dir = temp_dir
    before = {
        "level1": {
            "level2a": {"foo": 20, "bar": 21},
            "level2b": {"foo": 22, "level3": {"foo": 30}},
            "foo": 10,
        },
        "foo": 1,
    }

    # Write example yaml to temp_dir
    with open(output_dir / "example.yaml", "w", encoding="utf8") as file_h:
        yaml.safe_dump(before, file_h)

    # multiply level3 foo by 10
    after_1 = {
        "level1": {
            "level2a": {"foo": 20, "bar": 21},
            "level2b": {"foo": 22, "level3": {"foo": 300}},
            "foo": 10,
        },
        "foo": 1,
    }

    # multiply level2b foo by 10
    after_2 = {
        "level1": {
            "level2a": {"foo": 20, "bar": 21},
            "level2b": {"foo": 220, "level3": {"foo": 30}},
            "foo": 10,
        },
        "foo": 1,
    }
    return before, after_1, after_2


class TestCombineCSVFiles:
    """Methods to test the combine_csv method in utils.py."""

    def test_combine_csv_files(self, temp_dir):  # pylint: disable=W0621
        """Ensure combine_csv creates the correct csv file."""
        output_file = temp_dir / "combined.csv"
        utils.combine_csvs(temp_dir, output_file)

        assert output_file.exists()
        combined_csv = pd.read_csv(output_file)
        assert len(combined_csv) == 6

        assert len(combined_csv.columns) == 2

    def test_invalid_directory(self, temp_dir):  # pylint: disable=W0621
        """Ensure FileNotFoundError is raised if directory is not found."""
        with pytest.raises(FileNotFoundError):
            output_file = temp_dir / "combined.csv"
            utils.combine_csvs("invalid_directory", output_file)


class TestUpdate:
    """Methods to test the update function in utils.py."""

    def test_load_change_write(self, temp_dir, example_dicts):  # pylint: disable=W0613,W0621
        """Ensure loading a yaml, changing it, and writing it works correctly."""
        params = rre.load_yaml_file(temp_dir / "example.yaml")
        change_key_path1 = "level1.level2b.level3.foo"
        utils.update(params, change_key_path1, value=300)
        rre.write_to_yaml(params, temp_dir / "example1.yaml")
        should_be1 = params

        params2 = rre.load_yaml_file(temp_dir / "example.yaml")
        change_key_path2 = "level1.level2b.foo"
        utils.update(params2, change_key_path2, value=220)
        rre.write_to_yaml(params2, temp_dir / "example2.yaml")
        should_be2 = params2

        with open(temp_dir / "example1.yaml", encoding="utf8") as file1:
            actual1 = yaml.safe_load(file1)

        with open(temp_dir / "example2.yaml", encoding="utf8") as file2:
            actual2 = yaml.safe_load(file2)

        assert actual1 == should_be1
        assert actual2 == should_be2


class TestItemAndInterfaceCounter:
    """Methods to test the ItemAndInterfaceCounter class in more_utils.py."""

    def test_item_and_interface_counter_is_consistent(self):
        """Adding N ItemAndInterfaceCounter instances should have the same result as multiplying one of them by N."""
        u = utils.ItemAndInterfaceCounter.from_single_item("u")
        v = utils.ItemAndInterfaceCounter.from_single_item("v")
        u_plus_v = 5 * u + 2 * v
        multiply = 3 * u_plus_v
        add = u_plus_v + u_plus_v + u_plus_v
        assert multiply.items == add.items
        assert multiply.interfaces == add.interfaces

    def test_item_and_interface_counter_counts_correctly(self):
        """ItemAndInterfaceCounter should aggregate items and interfaces correctly."""
        u = utils.ItemAndInterfaceCounter.from_single_item("u")
        v = utils.ItemAndInterfaceCounter.from_single_item("v")
        x = utils.ItemAndInterfaceCounter.from_single_item("x")
        y = utils.ItemAndInterfaceCounter.from_single_item("y")
        z = utils.ItemAndInterfaceCounter.from_single_item("z")
        u_plus_v = u + v
        xyz_combo = 3 * x + 5 * y + 7 * z
        result = 11 * u_plus_v + 2 * xyz_combo + 13 * u_plus_v

        should_be_items = {"u": 24, "v": 24, "x": 6, "y": 10, "z": 14}
        should_be_interfaces = {
            ("u", "v"): 24,
            ("v", "u"): 22,
            ("x", "x"): 4,
            ("y", "y"): 8,
            ("z", "z"): 12,
            ("x", "y"): 2,
            ("y", "z"): 2,
            ("z", "x"): 1,
            ("v", "x"): 1,
            ("z", "u"): 1,
        }
        actual_items = result.items
        actual_interfaces = result.interfaces

        assert actual_items == should_be_items
        assert actual_interfaces == should_be_interfaces
