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
**Module** ``rigetti_resource_estimation.utils.py``

Some helper tools for performing resource estimations and analyzing RRE's results.
"""
from typing import Union, Any, Dict, Tuple
from pathlib import Path
import pandas as pd


def update(dictionary: Dict[str, Any], path: str, value: Any) -> None:
    """Update a dictionary's value at the key specified by a period separated string.

    For example, update(this_dict, "a.b.c", 10) would update this_dict, in place, by setting
    this_dict["a"]["b"]["c"] = 10.

    :param dictionary: dictionary to be updated.
    :param path: path to key, specified as a period separated string.
    :param value: new value of the dictionary at the key specified by path.
    """
    keys = path.split(".")
    current = dictionary
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def combine_csvs(csv_path: Union[str, Path], csv_out_name: str) -> None:
    """Concatenate all csv files in the csv_path directory.

    :param csv_path: path to directory containing result csvs.
    :param csv_out_name: file name for the combined csv file.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError
    csv_paths = csv_path.glob("**/*.csv")
    files = [file for file in csv_paths if file.is_file()]
    dfs = [pd.read_csv(file, index_col=False) for file in files]
    df = pd.concat(dfs)
    df.to_csv(csv_path / csv_out_name, index=False)


class ItemAndInterfaceCounter:
    """A class that arithmetically tracks frequency of objects and neighbour pairs from sequences when concatenated.

    For example,
    ```
    u = ItemAndInterfaceCounter.from_single_item('U')
    v = ItemAndInterfaceCounter.from_single_item('V')
    result = 3 * u + 5 * v   # denotes concatenating 3 U objects, 5 V objects, and then concatenating those results
    ```
    Then, `result.items` would give `{'U': 3,'V' :5}` and `result.interfaces` would give
    `{('U','U'):2,('U','V'):1,('V','V'):4,('V','U'):0}`
    """

    def __init__(
        self,
        items: Dict[Any, int] = None,
        interfaces: Dict[Tuple[Any, Any], int] = None,
        left_item: Any = None,
        right_item: Any = None,
    ) -> None:
        """
        :param items: a dictionary mapping the objects and their frequency (count).
        :param interfaces: a dictionary mapping pairs of objects and the frequency they occur beside each other.
        :param left_item: the left most object in the sequence.
        :param right_item: the right most object in the sequence.
        """
        self.items = items
        self.interfaces = interfaces
        self.left_item = left_item
        self.right_item = right_item

    @classmethod
    def from_single_item(cls, item: Any) -> "ItemAndInterfaceCounter":
        return cls(items={item: 1}, interfaces={}, left_item=item, right_item=item)

    def __mul__(self, factor):
        if not isinstance(factor, int) or factor <= 0:
            raise TypeError("Can only multiply by positive integer.")
        current_items = self.items.copy()
        current_interfaces = self.interfaces.copy()
        items = {key: value * factor for key, value in current_items.items() if value > 0}
        interfaces = {key: value * factor for key, value in current_interfaces.items() if value > 0}
        val = interfaces.get((self.left_item, self.right_item), 0) + factor - 1
        if val > 0:
            interfaces[(self.right_item, self.left_item)] = (
                interfaces.get((self.right_item, self.left_item), 0) + factor - 1
            )
        return ItemAndInterfaceCounter(items, interfaces, self.left_item, self.right_item)

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __add__(self, other):
        items = self.items.copy()
        interfaces = self.interfaces.copy()

        for item, repeats in other.items.items():
            items[item] = items.get(item, 0) + repeats
        for interface, repeats in other.interfaces.items():
            interfaces[interface] = interfaces.get(interface, 0) + repeats
        interfaces[(self.right_item, other.left_item)] = interfaces.get((self.right_item, other.left_item), 0) + 1
        return ItemAndInterfaceCounter(items, interfaces, self.left_item, other.right_item)
