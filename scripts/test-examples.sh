#!/bin/bash

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

set -o errexit
set -o nounset
set -o pipefail

script_directory="$(dirname "$0")"
pushd "${script_directory}/../examples" > /dev/null

package=$(poetry version | awk '{print $1;}')
version=$(poetry version | awk '{print $2;}')
include=${package//-/_}

echo
echo "Testing example Python notebooks for $package $version ($include)."
echo

if test -n "$(find . -maxdepth 1 -name "*.ipynb" -print -quit)"
then
  for notebook in *.ipynb; do
    echo "Run: poetry run papermill $notebook papermill/$notebook"
    echo
    mkdir -p papermill
    poetry run papermill $notebook papermill/$notebook
    echo
  done
else
  echo "Nothing to do."
  echo
fi

popd > /dev/null
