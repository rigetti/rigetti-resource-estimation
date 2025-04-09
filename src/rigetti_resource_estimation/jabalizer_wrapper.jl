#!/usr/bin/env julia

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

# A julia wrapper script to manage RRE's default FT-compiler, Jabalizer.
#
# Parts of the following code are inspired and/or customized from the original templates in the 
# open-source references of:
# [1] https://github.com/QSI-BAQS/Jabalizer.jl
# [2] https://github.com/zapatacomputing/benchq

# [1] is distributed under the MIT License and includes the following copyright and permission 
# statements:
# Copyright (c) 2021 Peter Rohde, Madhav Krishnan Vijayan, Simon Devitt, Alexandru Paler.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


if !isinteractive()
    import Pkg
    Pkg.add(name="JSON")
    Pkg.add(name="Jabalizer", version="0.5.2");
    Pkg.add("PythonCall"); Pkg.build("PythonCall");
end

using Jabalizer
using JSON


function run_jabalizer_mbqccompile(circuit, circuit_fname, suffix="", pcorrections_flag=false, debug_flag=false)
    temp_dir = tempdir();
    temp_qasm = joinpath(temp_dir, "temp-" * string(getpid()) * ".qasm");
    open(temp_qasm, "w") do file
        write(file, circuit)
    end
    
    mkpath("output/$(circuit_fname)/")
    json_path = "output/$(circuit_fname)/$(circuit_fname)$(suffix)_all0init_jabalizeframes.json"

    if debug_flag
        @info "\nRRE: Jabalizer FT-compiler starts for $(circuit_fname)$(suffix) widget ...\n"
    end
    Jabalizer.mbqccompile(  # We always assume an all "000...0" initialization
        Jabalizer.parse_file(temp_qasm);
        pcorrections=pcorrections_flag,
        universal=true,
        ptracking=true,
        filepath=json_path, 
    )
    if debug_flag
        @info "RRE: Jabalizer FT-compiler was completed for $(circuit_fname)$(suffix) widget.\n"
    end
end


function get_mbqc_data_from_jabalizejson(mbqc_output_qasm, jabalize_json)
    testjson = JSON.parsefile(jabalize_json)
    # Convert pcorrs keys from String to Int
    testjson["pcorrs"] = Dict(parse(Int, k) => v for (k,v) in testjson["pcorrs"])
    Jabalizer.qasm_instruction(mbqc_output_qasm, testjson)
end