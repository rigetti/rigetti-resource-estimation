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
The root of the `rigetti_resource_estimation` (RRE) package, which provides framework foundations and is accessible to
all layers.
"""

import logging
import sys
import re
from os import path
from typing import Union, Optional, Literal, Dict, Any
from pathlib import Path

import yaml

if sys.version_info < (3, 8):
    from importlib_metadata import version  # pragma: no cover
else:
    from importlib.metadata import version


__version__ = version(str(__package__))


PARAMS_YAML_FILENAME = "params.yaml"


def check_for_qasm(filepath: Union[str, Path]) -> bool:
    """Check if `filepath` holds a valid OPENQASM2.0 file."""
    check = False
    try:
        with open(filepath, "r", encoding="utf8") as file:
            circ_content = file.readlines()
            qasm_pattern0 = re.search(r"OPENQASM 2.0;", circ_content[0])
            qasm_pattern1 = re.search(r"include \"qelib1.inc\";", circ_content[1])
        if qasm_pattern0 is not None and qasm_pattern1 is not None:
            check = True
    except IndexError:
        logging.info("RRE could not find a valid OPENQASM2.0 pattern in the inputted circuit; moving on!\n")
    return check


def load_yaml_file(filepath: Optional[Union[str, Path]] = None) -> dict:
    """Read a YAML file and parse it into a dict.

    :param filepath: path to the .yaml file.

    :returns: A dict version of the YAML input.
    """
    if filepath is None:
        filepath = path.join(path.dirname(path.realpath(__file__)), PARAMS_YAML_FILENAME)

    with open(filepath, "r", encoding="utf8") as yaml_stream:
        parsed_yaml = yaml.safe_load(yaml_stream)

    return parsed_yaml


def write_to_yaml(dict_for_yaml: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Write dict to a .yaml file.

    :param dict_for_yaml: dictionary to write to YAML.
    :param filepath: location to write the .yaml file to.
    """
    with open(filepath, "w", encoding="utf8") as yaml_file:
        yaml.safe_dump(dict_for_yaml, yaml_file)


class Configuration:
    """Class that holds hardware and logical configurations as well as convenience methods for accessing them."""

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Store parameters from the passed params dict.

        :param params: dictionary of parameters for use in resource estimation.
        """
        self.params = params

    @property
    def decomp_method(self) -> str:
        """Return the decomp method set in the params dict."""
        return self.params["1q_unitary_decomp_method"]

    @property
    def fixed_eps(self) -> float:
        """Return the eps value set in the params dict."""
        return self.params["fixed_eps"]

    @property
    def ftqc_params(self) -> dict:
        """Return the ftqc parameters set in the params dict."""
        return self.params["ftqc"]

    @property
    def widget_params(self) -> dict:
        """Return the widget parameters set in the params dict."""
        return self.params["distill_widget_table"]

    @property
    def target_p_algo(self) -> float:
        """Return the total failure probability budget, `target_p_algo`, set in the params dict."""
        return self.params["target_p_algo"]

    @property
    def p_gates(self) -> float:
        """Return the physical gate error rate set in the params dict."""
        return self.params["phys_gate_error_rate"]

    @property
    def decoder_char_timescale_sec(self) -> float:
        """Return the decoder characteristic timescale set in the params dict."""
        return self.params["decoder_char_timescale_sec"]

    @property
    def error_scaling_coeffs(self) -> Dict[str, float]:
        """Return the coefficients of the power-law physical error to surface code error scaling."""
        return self.params["error_scaling_coeffs"]

    @property
    def n_steps(self) -> Union[Literal["from_filename"], int]:
        """Return number of times the same input circuit is meant to be consecutively executed on the FTQC."""
        return self.params["n_steps"]
