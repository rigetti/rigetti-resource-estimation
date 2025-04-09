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

# GLOBALS
SHELL := bash
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

# COMMANDS
.DEFAULT := help


.PHONY: help
help:
	@printf 'Supported Makefile commands:\n\n'
	@fgrep -h "##" Makefile | fgrep -v fgrep | sed -e 's/:.*##/: /' | sed -e 's/##//' | sed -e '/ :/ s/^/    /'
	@printf '\n'

.PHONY: check-poetry
check-poetry:  ## Check if poetry is available in the current shell.
	@if ! command -v poetry &> /dev/null; then echo "Missing Poetry! (https://python-poetry.org/)"; \
		exit 1; fi

.PHONY: poetry-install
poetry-install:  check-poetry  ## Install into a Poetry virtual environment (quiet mode).
	@poetry install

.PHONY: poetry-install-quiet
poetry-install-quiet: check-poetry  ## Install into a Poetry virtual environment (quiet mode).
	@poetry install -q

.PHONY: check-style  
check-style: poetry-install-quiet  ## Check conformance to code style rules.
	@poetry run ruff check .

.PHONY: check-format  
check-format: poetry-install-quiet  ## Check conformance to code format rules.
	@poetry run black --check --diff .

.PHONY: lint
lint: poetry-install-quiet  ## Make automatic updates to code style and format.
	@poetry run black .
	@poetry run ruff check --fix .

.PHONY: build-docs
build-docs: poetry-install-quiet  ## Build the project documentation; HTML format.
	@./scripts/build-docs.sh

.PHONY: build-package
build-package: poetry-install-quiet  ## Build the project package's sdist and wheel.
	@poetry build

.PHONY: test-package
test-package: poetry-install-quiet  ## Test the package; includes a coverage report.
	@./scripts/test-package.sh

.PHONY: test-examples
test-examples: poetry-install-quiet  ## Test all notebooks in examples/; stores the notebook output.
	@./scripts/test-examples.sh

.PHONY: run-examples
run-examples: poetry-install-quiet  ## Estimate resources for QASMs at examples/input.
	@./scripts/run-examples.sh

.PHONY: jupyter-lab
jupyter-lab: poetry-install-quiet  ## Run a Jupyter server and launch the lab view.
	@poetry run jupyter lab
