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
**Module** ``rigetti_resource_estimation.analysis``

This module provides additional tools for performing analysis of resource estimations done through the RRE pipeline.
"""
import os
import shutil
import logging
from pathlib import Path
import click
import rigetti_resource_estimation
from rigetti_resource_estimation import more_utils
from rigetti_resource_estimation.estimation_pipeline import estimation_pipeline

logging.basicConfig(level=logging.INFO)


class SweepController:
    """Class to process and hold the values required to perform a parameter sweep for resource estimations."""

    ESTIMATE_PIPELINE_CMD = ["poetry run python src/rigetti_resource_estimation/estimation_pipeline.py"]
    PARAMS_FILE = "params.yaml"
    BACKUP = "params.bk"
    OUTPUT_ARG = "--output-csv"
    COMBINED_CSV_NAME = "combined.csv"
    PARAMS_DIR = Path("src/rigetti_resource_estimation")

    def __init__(self, param: str, values: str, circ_path: str, output_csv: str) -> None:
        """
        Initialize some common parameters.

        :param param: parameter that will be swept over.
        :param values: values, separated by commas, that will be assigned to the parameter during sweep.
        :param circ_path: path to qasm file to run the resource estimation on.
        :param output_csv: path to output .csv, and filename template for the output CSV.
        """
        logging.info("Preparing sweep controller ....")
        self.values = values.split(",")
        self.circ_path = circ_path
        self.output_csv = output_csv
        self.param = param
        self.output_filepath = Path(self.output_csv)
        self.output_dir = self.output_filepath.parent
        self.file_name = self.output_filepath.stem
        logging.info("Done.")

    @property
    def graph_state_opt(self):
        """Return the flag for the `graph_state_opt` depending on the existence of the 'output/' path."""
        if Path("output/").exists():
            return "resume"
        return "save"

    @property
    def param_key_path(self):
        """Split param name and return a list representing key path to parameter for use in a nested dictionary."""
        return self.param.split(".")

    @property
    def param_for_path(self):
        """Return a representation of the parameter suitable for a file path."""
        return self.param.replace(".", "-")

    @property
    def value_path_pairs(self):
        """Return a list of tuples containing parameter values and corresponding output file paths."""
        file_path = self.output_filepath
        pairs = []
        for val in self.values:
            new_path = file_path.with_name(f"{self.file_name}_{self.param_for_path}{val}{file_path.suffix}")
            pairs.append((float(val), new_path))
        return pairs

    @property
    def params_path(self):
        """Return the full path to the params.yaml file."""
        return self.PARAMS_DIR / self.PARAMS_FILE

    @property
    def backup_path(self):
        """Return the full path to the backup params file."""
        return self.PARAMS_DIR / self.BACKUP


def perform_sweep(parameter: str, values: str, circ_path: str, output_csv: str) -> None:
    """Perform the sweep over values for a specific parameter, save the resulting CSVs and combine them.

    :param parameter: parameter to sweep over. Should be of form 'key1.key2.key3.key4' where the '.' identifies level in
        dict structure. So in this example, the parameter to sweep over is key4, but you need to descend 3 levels in the
        nested dictionary to arrive there.
    :param values: values for the parameter to sweep over.
    :param circ_path: path to the input circuit file to run the resource estimation on.
    :param output_csv: path to the output CSV and a template to save them.
    """
    # Preparing the sweeper
    logging.info("Starting parameter sweeper ...")
    sweep_cntl = SweepController(parameter, values, circ_path, output_csv)

    logging.info("Backing up parameter file %s (%s)", sweep_cntl.params_path, os.getcwd())
    shutil.copy2(sweep_cntl.params_path, sweep_cntl.backup_path)  # backup params.yaml

    logging.info("Preparing output directory at %s", str(sweep_cntl.output_dir))
    sweep_cntl.output_dir.mkdir(parents=True, exist_ok=True)

    # loop over parameter values, run the process, and store results
    logging.info("Beginning the sweep ...")
    logging.info("Sweeping over %s", sweep_cntl.param)
    for val, path in sweep_cntl.value_path_pairs:
        logging.info("    Running parameter value %s", val)
        output_path = str(path)
        params = rigetti_resource_estimation.load_yaml_file(sweep_cntl.params_path)
        more_utils.update(params, sweep_cntl.param, val)
        config = rigetti_resource_estimation.Configuration(params)

        # Create the cli command and run it
        logging.info("Running resource estimator ....")
        estimation_pipeline(
            circ_path=sweep_cntl.circ_path,
            output_csv=output_path,
            graph_state_opt=sweep_cntl.graph_state_opt,
            config=config,
        )

    logging.info("Finished parameter sweep.")

    # Create final output csv
    logging.info("Combining CSVs ...")
    more_utils.combine_csvs(sweep_cntl.output_dir, sweep_cntl.COMBINED_CSV_NAME)
    logging.info("Done. Combined results at %s", sweep_cntl.output_dir / sweep_cntl.COMBINED_CSV_NAME)

    # Restore params.yaml and remove backup
    logging.info("Cleaning up ...")
    logging.info("Restoring parameters file, %s", sweep_cntl.params_path)
    shutil.copy2(sweep_cntl.backup_path, sweep_cntl.params_path)
    logging.info("Removing backup parameters file ...")
    os.remove(sweep_cntl.backup_path)

    logging.info("Parameter sweep completed successfully! Please enjoy your new data responsibly.")


@click.group()
def cli():
    """Entry point to the analysis.py cli."""
    logging.info(
        "Welcome to the parameter sweep analysis tool based on Rigetti Resource Estimation version %s",
        rigetti_resource_estimation.__version__,
    )


@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument("parameter", type=str)
@click.argument("values", type=str)
@click.option("--circ-path", type=click.Path(), help="Path to the qasm file to run the resource estimation on")
@click.option("--output-csv", type=str, help="Path to the output CSV and a template to save them")
def sweep(parameter: str, values: str, circ_path: str, output_csv: str) -> None:
    """Sweep over values for a specific parameter, save the resulting CSVs, and combine them.

    For example, executing:

    ```poetry run RREsweep ftqc.inter_magictransfer_timescale_sec 2e-6,3e-6,4e-6,6e-6 --qasm-path qasms/qft3.qasm
    --output-csv output/qft_tock_runs.csv```

    will run the sweep over values for `inter_magictransfer_timescale_sec` param (nested below the ftqc key) of 2e-6,
    3e-6, 4e-6, and 6e-6. CSV files will be saved in output/ with file names `qft_tock_runs.csv` but modified to look
    like `qft_tock_runs_ftqc-inter_magictransfer_timescale_sec_2e-6.csv` (for example). The final combined CSV file will
    be located in output/ with file `combined.csv`.

    :param parameter: parameter to sweep over. Should be of form 'key1.key2.key3.key4' where the . identifies level in
        dict structure. So in this example, the parameter to sweep over is key4, but you need to descend 3 levels in the
        nested dictionary to arrive there.
    :param values: values for the parameter to sweep over.
    :param circ_path: path to qasm file to run the resource estimation on.
    :param output_csv: path to output .csv, and filename template for output CSV.
    """
    perform_sweep(parameter, values, circ_path, output_csv)


if __name__ == "__main__":
    cli()
