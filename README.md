Copyright
---------

Copyright 2022-2024 Rigetti & Co, LLC

This Computer Software is developed under Agreement HR00112230006 between Rigetti & Co, LLC and
the Defense Advanced Research Projects Agency (DARPA). Use, duplication, or disclosure is subject
to the restrictions as stated in Agreement HR00112230006 between the Government and the Performer.
This Computer Software is provided to the U.S. Government with Unlimited Rights; refer to LICENSE
file for Data Rights Statements. Any opinions, findings, conclusions or recommendations expressed
in this material are those of the author(s) and do not necessarily reflect the views of the DARPA.

Use of this work other than as specifically authorized by the U.S. Government is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and limitations under
the License.


# Rigetti Resource Estimation

The [Rigetti Resource Estimation](https://github.com/rigetti/rigetti-resource-estimation) (RRE) is a resource
estimation tool for future fault-tolerant superconducting quantum hardware based on logical, physical-level inputs and
graph state compilation to measurement-based quantum computing and surface code operations (see 
[PhysRevA 73 022334](https://doi.org/10.1103/PhysRevA.73.022334), 
[Quantum Sci. Technol. 2 025003](https://doi.org/10.1088/2058-9565/aa66eb), and
[arXiv:2209.07345](https://arxiv.org/abs/2209.07345)). The program is written in Python and supports a variety of
estimation methods.

## Installation

RRE can be installed and used as a stand-alone tool from the source.

### Prerequisites

#### Poetry

You will need to have [Poetry](https://python-poetry.org/docs/) installed on your system to install the RRE
tool from the source. The current official installer can be invoked for Linux using the following command:

```
curl -sSL https://install.python-poetry.org | python3 -
```

#### Julia and Jabalizer graph-state-compilation library 

RRE uses [Jabalizer](https://github.com/QSI-BAQS/Jabalizer.jl) as its default fault-tolerant compiler for all
graph-state-based resource estimates. Jabalizer will be installed (through Julia's official packages only on the first
run) and invoked upon each run of RRE when the user selects the `jabalizer` estimation method. We provide a
[jabalizer_wrapper.jl](src/rigetti_resource_estimation/jabalizer_wrapper.jl) script that automatically registers
dependent Julia libraries, configures, and runs Jabalizer.

You must have [Julia](https://julialang.org/) installed on your system to run Jabalizer. We recommend installing Julia
using [Juliaup](https://github.com/JuliaLang/juliaup); for Linux installations execute the following command:

```
curl -fsSL https://install.julialang.org | sh -s -- -y
```

### Source install

You will need to download the repository source using the GitHub download feature or cloning the repository
using Git. Once you have done this, you will have a local copy of the source under `rigetti_resource_estimation`.

Install the Python package and its dependencies using Poetry by executing the following command:

```
poetry install
```


## Getting started

Our front-end resource estimation pipeline is in
[estimation_pipline.py](src/rigetti_resource_estimation/estimation_pipeline.py) and can either be run as a command-line
tool or directly as a library dependency via the `estimation_pipeline()` function. Try to run our command-line tool
using the following command, which will display help on all options:

```
poetry run python src/rigetti_resource_estimation/estimation_pipeline.py --help
```

Our tool includes a default four-qubit Quantum Fourier Transform OpenQASM 2.0 example circuit that you may try and
edit as desired. To run this example with the default Jabalizer graph-state compiler and log the results to the console, 
run the following command:

```
poetry run python src/rigetti_resource_estimation/estimation_pipeline.py --log="INFO" 
```

A complete example would be to estimate resources for a circuit of your choice by specifying the QASM input file,
with the results appended to a CSV file and the intermediate adjacency list of the compiled graph state also saved for
characteristic analysis. Try the following command as an example and tailor it to your needs:

```
poetry run python src/rigetti_resource_estimation/estimation_pipeline.py --log "DEBUG" \
--circ-path examples/input/qft4.qasm --est-method "jabalizer" --output examples/test.csv \
--graph-state-opt "save"
```

A possible use of our tool is as a design aid, sweeping specified fields from the `params.yaml` configuration file over
ranges of values to understand how hardware design changes might impact final resource requirements. An example of this
where we inspect the impact of varying the time needed for an inter-module surface-code "tock" is:

```
poetry run sweep ftqc.intermodule_tock_sec "2e-6, 3e-6, 4e-6, 6e-6" \
--circ-path examples/input/qft4.qasm --output-csv examples/test.csv
```

We also have a Jupyter demo notebook with more examples, [demo.ipynb](python/notebooks/demo.ipynb), which you may find
helpful as a starting point for your analysis. You can launch Jupyter Lab using the following command and then navigate
to the "examples" directory to load this notebook:

```
make jupyter-lab
```

## Development

You can control typical development and testing tasks by the `Makefile`; you can view the list of options by running:

```
make
```

Formatting and style checks are run in CI, as are the package and demonstration notebook tests. We suggest you
lint your code before pushing by running:

```
make lint
```

All tests can be invoked with coverage enabled by running the command:

```
make test-package
```

## Common issues

### 1. I get errors similar to `FileNotFoundError: [Errno 2] No such file or directory`.

Check that you are invoking the command-line tool from the repository root and that all paths you have provided are
correct relative to the current working directory.

### 2. I get errors like `You are using both a virtual and conda environment, cannot figure out which to use!`.

You have both the `conda` and `virtualenv` Python environments activated, and `juliapkg` cannot decide which one to use.

One solution is to run `conda deactivate` several times, ensuring all Conda envs and vars are unset, and then switch
to a desired env for `poetry` either through the `poetry run` commands or in an interactive shell via `poetry shell`.
You could always use `poetry env use /full/path/to/python` to change the environment Poetry uses -- see
[here](https://python-poetry.org/docs/managing-environments/) for more details.  

### 3. I am seeing errors like `/.pyenv/versions/3.8.14/lib/libpython3.8.a could not be opened.`

This is an issue that occurs if you use `pyenv` to manage your `python` installs. `PythonCall` is trying to access the
shared libraries from your `python` install. This is a
[known problem.](https://github.com/JuliaPy/PythonCall.jl/issues/318). The fix is to reinstall `python`, via `pyenv`,
only by using the command `env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install <python version>`. In this way,
`pyenv` makes the shared libraries available to `PythonCall`. More information on this can be found
[here.](https://github.com/pyenv/pyenv/blob/master/plugins/python-build/README.md#building-with---enable-shared)
