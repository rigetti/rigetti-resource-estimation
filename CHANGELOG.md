Copyright
---------

Copyright 2022-2025 Rigetti & Co, LLC

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


# Changelog

### Version 0.26.32

* Fix minor documentation build error.

### Version 0.26.31

* Update copyright statements to prepare for public release.

### Version 0.26.30

* We have implemented a series of bug fixes, error fixes, and structural changes to make the software consistent with a new version of [RRE manuscript](https://arxiv.org/abs/2406.06015). These changes are primarily reflected in module [resources.py](src/rigetti_resource_estimation/resources.py). The following is a technical summary of the main changes:
  * The resource class `NumIONodes` was removed because in the fixed architecture and widgetization, we assume the number of input or output nodes for all widgets equals the number of algorithmic qubits in the input algorithm.
  * We have removed the tensor network decoder entries from all modules, replacing them with attributes of a generic GPU or FPGA-based decoder, which we believe is more suited for our superconducting architecture.
  * We made the order of appearance and some other details of outputs from [resources.py](src/rigetti_resource_estimation/resources.py) consistent with Appendix A of [RRE manuscript](https://arxiv.org/abs/2406.06015).

### Version 0.26.29

* We changed the way that widgets and stitches are obtained from a circuit dependency graph. They are now obtained using the new ItemAndInterfaceCounter class.
* We also added unit tests for the new ItemAndInterfaceCounter class

### Version 0.26.28

* Fix bug involving subcircuits with multiple moments during widgetziation. Also, fixed bug related to the ordering of leaf nodes when obtaining widgets and stitches.

### Version 0.26.27

* Updated widgetization to use prefab functions rather than prefab look-ups. This allows the user to use the properties of a gate to build their user-defined gate decompositions.  

### Version 0.26.26

* We have implemented a series of changes in [resources.py](src/rigetti_resource_estimation/resources.py) to improve the readability of output parameters. To highlight some of these changes:
  * We have added a series of new parameters to report the per-module and total number of logical and physical qubits in the main components of the intra-module architecture more clearly.
  * The deprecated column `transpiled_widget` was removed from CSV outputs.
  * We've updated the module to report the total no. of cryogenic modules rather than modules per leg of ladder-like macro-architecture.

### Version 0.26.25

* We updated WidgetizationResults to provide widgets_and_stitches output.
* Updated ordered_leaf_nodes to use edge weights and report as (node, repetitions) rather than a flat list of repeated nodes. 
* Add lru_caching for creating ordered_leaf_nodes.

### Version 0.26.24

* We updated WidgetizationResults to provide access to leaf nodes, ordered leaf nodes, root, node, and circuit reconstruction properties.
* Updates widgetization graph to be a networkx MultiDiGraph to allow for repeated edges. Also added 'idx' property on graph edges to track the order in which the edges appeared during decomposition. This maintains the time ordering of the subcircuits.

### Version 0.26.23

* We recasted WidgetizationResult as a class, rather than a dataclass, which takes a circuit decomposition graph from the widgetization process and processes it into different objects.
* We updated tests to reflect these changes to the WidgetizationResult definition.

### Version 0.26.22

* We updated identification of leaf nodes by graph properties alone (nodes that have out degree of 0)
* We added support for user defined decomposition of a gate before defaulting to using cirq decomposition methods.
* We also added new tests for widgetization focusing on correct graph creation and testing passing of user defined decomposition schemes.

### Version 0.26.21

* Removed leaf tracking on graph nodes when constructing circuit decomposition graph.

### Version 0.26.20

* We added boolean flags in graph state equivalence mappers to allow users to toggle on/off the use of gate names and/or qubits in assessing graph state equivalence of operations.

### Version 0.26.19

* We updated the "order" field formatting for `widgetized_circ_dict` to better capture the ordering and repetitions of widgets in the algorithm. The new format offers more brevity and user-friendliness as it only lists the number of times each widget and their unique crossing is repeated regardless of their position in the input algorithm (for the resource estimations, only the former matters). We have included some examples below.  
  * Example 1: 
  W1,W2,W1,W2,W1,W2,W2 -> widgetized_circ_dict = {"widgets": {"W1":(circ1, 3), "W2":(circ2, 4)}, "stitches":{("W1","W2"):3, ("W2","W1"): 2, ("W2","W2"): 1}}
  * Example 2: 
  W1,W2,W3,W1,W2,W3,W1,W2,W3 -> widgetized_circ_dict = {"widgets": {"W1":(circ1, 3), "W2":(circ2, 3), "W3":(circ3, 3)}, "stitches":{("W1","W2"):3, ("W2","W3"): 3, ("W3","W1"): 2}}
Here, `circ1` and `circ2` are the actual logical circuits for widgets labelled "W1" and "W2" expressed as `criq` objects.

### Version 0.26.18

* We extended the "order" field formatting for `widgetized_circ_dict` input to efficiently capture all possible ordering and repetitions of the complete algorithm. Below are some examples of how the new format works. For efficiency reasons, RRE will automatically estimate the space-time resources required for the complete algorithm by multiplying appropriate pre-factors on relevant parameters, as per patterns in "order", rather than explicitly forming the complete schedules.  
  Example 1: W1,W2,W1,W2,W1,W2,W2 -> "order": [{"pattern": {"W-1": 1, "W-2": 1}, "reps": 3}, {"pattern": {"W-2": 1}, "reps": 1}]
  Example 2: W1,W2,W3,W1,W2,W3,W1,W2,W3 -> "order": [{"pattern": {"W-1": 1, "W-2": 1, "W-3": 1}, "reps": 3}]

* We have implemented the following error fixes and changes in module [hardware_components.py](src/rigetti_resource_estimation/hardware_components.py):
  *  To calculate `n_seq_consump`, an error in the placements of Rz and T counts was fixed, and we introduced improved formulas in writing `n_max_nodes()` and `n2`.
  * All module crossing counts were updated to be upper bounded by "num_modules_per_leg - 1" as this is the maximum module crossing needed to interconnect two nodes on the same leg in the worst-case scenario. 

* We have implemented the following error fixes and changes in [hardware_components.py](src/rigetti_resource_estimation/hardware_components.py):
  *  Class `OverallPrepDelay` now correctly reports `consump_sched_times.intra_prep_delay_sec` as intended rather than mistakenly reporting `consump_sched_times.intra_distill_delay_sec`. 
  * In Class `ReqPhysicalQubits`, we now report the total number of allocated physical qubits for both legs of macro-architecture (a factor two increase compared to the previously reported total number of active qubits per leg).
   
### Version 0.26.17

* We added the ability for the default decomposers to decompose z rotation gate with an angle, very close to zero, into cirq.I. 

### Version 0.26.16

* We modified estimation pipeline to pass through per-widget t, z rotation, and clifford counts and then used these per-widget counts to parameterize cabaliser's max memory during compilation process. 

### Version 0.26.15

* We added reset functionality to the translators so that translators call counts are reset each time transpilation is called. We also updated estimation pipeline to sum counts from all widgets, not just new ones.

### Version 0.26.14

* We added seed to random circuits in widgetization tests to eliminate stochasticity.

### Version 0.26.13

* We added additional unit tests for `widgetization.py`.

### Version 0.26.12

* We added unit tests for `gs_equivalence.py`.
* We modified `GraphStateEquivalenceMapper` to use a dictionary to look up supported gates with a graph state equivalence definition and provide a default definition for gates not supported yet.
* We created a dictionary of currently supported gates for graph state equivalence including QubitizedRotation, PauliStringLCU, prepare_pauli_lcu, SelectPauliLCU (pyLIQTR), as well as Rx, Ry, Rz (cirq).

### Version 0.26.11

* We added unit tests for `translators.py`.

### Version 0.26.10

* We added unit tests for `decomposers.py`.

### Version 0.26.9

* We added support for passing a decomposition context when instantiating the CirqWidgetizer class, allowing the user to specify a qubit manager for widgetization. 
* We added support for passing a decomposition context when calling decompose_and_qbmap(), allowing the user to specify a qubit manager for full decomposition of widgets. 
* We added support for passing a decomposition context when calling estimation_pipeline(), allowing the user to specify a qubit manager for full decomposition of widgets during the estimation pipeline process. 

### Version 0.26.8

* We removed full decomposition step in widgetization. Widgets are now returned without being fully decomposed.
* We removed the circuit santitation step from widgetization with decomposers fully handling any gate modification or manipulation pre-transpilation.

### Version 0.26.7

* We modified the arbitrary rotation Cirq to Cabaliser translator to also act on qualtran zpower gates.

### Version 0.26.6

* We modified the `estimation_pipeline` method to calculate number of input qubits AFTER it fully decomposes the widgets, not directly from the widgets themselves.

### Version 0.26.5

* We updated CIRQ_QT_BLOQ_DECOMP decomposer to handle basic bloqs TGate, CNOT, XGate, YGate, ZGate, SGate.

### Version 0.26.4

* We have added a new test module, [test_unitary_verif_cabaliser.py](tests/verify_n_validate/test_unitary_verif_cabaliser.py). The module provides a unit test to verify the unitariness of graph states and schedules outputted by RRE and its default FT-compiler, Cabaliser, robustly but only for small sizes of input qubits (exact simulations are performed). See the module docs for further details.

### Version 0.26.3

* We added in new operation translators to handle the inverse of self-inverse gates.

### Version 0.26.2

* We have added an intercepting decomposer to handle qualtran's AddConstantMod and ModAddK operations.

### Version 0.26.1

* We have fixed a bug in [cabaliser_wrapper.py](src/rigetti_resource_estimation/cabaliser_wrapper.py) module relating to Cabaliser's angle tags, which was causing all measurement tags to print as zeros incorrectly.

### Version 0.26.0

* We have integrated [Cabaliser](https://github.com/Alan-Robertson/cabaliser) as a submodule and the default FT-compiler for RRE, which replaces Jabalizer. Our extensive benchmarks have indicated Cabaliser is one to two orders of magnitude faster and more memory efficient, mainly due to employing input streaming, avoiding unnecessary disk reading and writing, sub-widgetization/sequencing, and multi-processing techniques -- See Cabaliser's docs for more details. The following changes were required as part of Cabaliser integration:
  * A set of new and modular transpilation classes were added to the [transpile.py](src/rigetti_resource_estimation/transpile.py), [translators.py](src/rigetti_resource_estimation/translators.py), and [decomposers.py](src/rigetti_resource_estimation/decomposers.py) modules to pre-process (parsing, decomposing unsupported, and counting gates for) the input widgets before sending to Cabaliser. 
  * RRE now only reads the input logical algorithm from a custom dictionary that supports widgetizations and provides the subcircuits in Google's cirq format. To familiarize yourself with the new dictionary input, please take a look at the examples at the top of the [estimation_pipeline.py](src/rigetti_resource_estimation/estimation_pipeline.py) module. In particular, we have temporarily deprecated the support for qasm inputs due to disk reading and writing inefficiency. In future commits, we plan to re-add support for OpenQASM2.0 and other popular quantum logical instruction formats.
  * The Cabaliser core is written in C for high efficiency, with some Python wrapping modules available (we also provide our own wrapper module as [src/rigetti_resource_estimation/cabaliser_wrapper.py](src/rigetti_resource_estimation/cabaliser_wrapper.py)). Therefore, users must perform extra C-libraries build steps as part of the RRE installation from the source. Please have a look at our updated README for details.

### Version 0.25.0

* We have modified RRE to work with an updated JSON format for widgetized input algorithms. The updated format features fields specifying unique widgets and lists the time ordering in which they will be repeated. Digesting from this new format would significantly speed up FT compilations as RRE no longer needs to compile all subcircuits, only the unseen widgets for once. Initial small-scale testing proved to be an order of magnitude improvement in the compilation speed and the size of RRE custom JSON input circuits.

### Version 0.24.0

* We have implemented a new macroscopic-level FT architecture as a simple extension of the two-fridge design. The new architecture consists of a ladder of many fridges, where each leg does interleaving graph preparation and consumption stages as before. Given large input algorithms, the architecture can now scale up arbitrarily large in the space direction with as many fridge pairs on the ladder as required to fit the quantum bus. In other words, when the logical size of the quantum bus is more demanding than a singular module space, RRE will no longer error out and fit the time-sliced (sub)graphs across many fridges as if there were a single larger module -- some operations will become inter-fridge ones. The intra-module micro-architecture remains unchanged. Details of the new architecture can be found in the RRE manuscript [arXiv:2406.06015](https://arxiv.org/abs/2406.06015).

* We provide an improved micro-architecture for the FTQC with T-state queueing, parallelization, and explicit component allocation. This updates the methods for calculating the code distance (parameter d in [estimate_from_graph.py](src/rigetti_resource_estimation/estimate_from_graph.py)) and time results (in [hardware_components.py](src/rigetti_resource_estimation/hardware_components.py)). We've included more details below. 
  * In [estimate_from_graph.py](src/rigetti_resource_estimation/estimate_from_graph.py), we previously estimated d by approximately and iteratively solving the logarithmic Equation (15) of the RRE manuscript v1 [arXiv:2406.06015v1](https://arxiv.org/abs/2406.06015/v1). We assumed a fully sequential T-state consumption: only one T-state was available for consumption at a given time, even if many T-factories were allocated within our intra-module architecture. This would lead to valid but worst-case-scenario d-values, i.e., the largest "safe" code distance meeting all architectural assumptions. This version detailed a parallelized T-state-queueing intra-module architecture (see below) and implemented a modified iterative and exact `d_from_rates()` finder. We assumed we could generate and consume as many simultaneous T-states as there are T-factories in a fridge. This leads to finding considerably lower optimal d, reducing most FT resources. Note that the architectural assumptions, including error rate conditions and validating preparation and consumption of T-states for all timesteps, are checked in [hardware_components.py](src/rigetti_resource_estimation/hardware_components.py) and [estimate_from_graph.py](src/rigetti_resource_estimation/estimate_from_graph.py) for any proposed d.
  * The following briefly describes each component in the new intra-module architecture: Each fridge is considered a rectangular grid of logical qubits/patches as the building block of all components. The width and length are set to be as close to each other as possible, and every patch contains 2*d*d physical qubits. Other than unallocated, wasted space, all fridges have four main components. The first two components come from the bilinear quantum bus of ancilla and graph-state/data logical qubits. We lay these out in a comb or snake-like pattern to fill in the longer side of the fridge. Third is the linear T-transfer bus; the last components are T-distillation widgets/factories. On the remaining side of the module, the T-transfer-bus sandwiches as many T-factories as possible and is laid out in a comb-like pattern again so that two columns or more must touch the long side of the ancilla quantum bus. The T transfer bus actively stores T-states from factories and queues them during all sub-schedules. As detailed below, the T-transfer bus can feed many parallel T-states to the quantum bus during the subgraph consumptions. For full details, see the RRE manuscript [arXiv:2406.06015](https://arxiv.org/abs/2406.06015).

### Version 0.23.6

* Change log and packaging updates to prepare for public release.

### Version 0.23.5

* Fixed hard-coded mention of `params.yaml` in console reporting.

### Version 0.23.4

* Add passthrough of circuit decomposition information to output.

### Version 0.23.3

* When resource estimation fails due to insufficient physical resources, zero resources are now reported and estimation completes successfully. Previously, an error would be raised which would prevent batch execution over many circuits.

### Version 0.23.2

* Documentation improvements.

### Version 0.23.1

* We have improved how distillation and consumption times are calculated in [hardware_components.py](src/rigetti_resource_estimation/hardware_components.py) based on the following:
  * The T-gates from arbitrary Rz decompositions cannot be parallelized for consumption measurements.
  * The pre-factor of 8 in a tock of surface code cycle should also be included in `t_intra_magicstates` calculations.

### Version 0.23.0

* We have switched to a bilinear-bus, two-fridge architecture for the FTQC by default due to better performance in the output time resources. See [hardware_components.py](src/rigetti_resource_estimation/hardware_components.py) for details.

* We have integrated a new version of Jabalizer, v0.5.2.

### Version 0.22.0

* We have integrated a new version of Jabalizer, v0.5.0, with built-in Pauli Tracker support, universal graph output, and stitching capabilities.

### Version 0.21.2

* We have added independently configurable T-state transfer and graph handover characteristic times to the example config file, [params.yaml](src/rigetti_resource_estimation/params.yaml), plus some other minor modifications. These changes allow users to switch architectures from "tri-fridge with T-distilleries separated" to "single or two-fridge with T-distilleries inside" by setting appropriate config parameters.

* More output parameters were added to RRE's `stdout` and default CSV.

### Version 0.21.1

* Improved [README.md](README.md).
* Fixed a bug in the calculations of hardware times in [hardware_components.py](src/rigetti_resource_estimation/hardware_components.py).

### Version 0.21.0.1

* Updated CI and Makefile to build Python package wheel and capture this and documentation as a build artifact.

### Version 0.21.0

* We have integrated the new versions of [Pauli Tracker](https://github.com/QSI-BAQS/pauli_tracker.git) and [MBQC Scheduling](https://github.com/taeruh/mbqc_scheduling.git) libraries, now with PyPI releases.

### Version 0.20.0

* We have deprecated hard-coded QSP building blocks format for decomposed circuits, which was limited only to specific QSP-style QASM inputs. In its place:  
  * We have introduced a flexible 'RRE custom JSON with decomposed QASMs' standard. The new JSON format can input QASM strings from serial circuit decomposition, their ordering, and other metadata.
  * An example of such a JSON file can be found in "./examples/input/qft4-decomposed3.json".

* For decomposed input circuits, the substrate scheduling is now performed on each time-sliced block, and the full schedule is generated by concatenating these sub-schedules in serial assuming an interleaving architecture.

* We have adopted a new bilinear-bus tri-fridge architecture for the resource estimations; see [hardware_components.py](src/rigetti_resource_estimation/hardware_components.py).
  * The new architecture includes two interconnected fridge modules for algorithmic time-ordered subcircuit widgets. These run interleaving subgraphs. One module is growing the graph, i.e., by preparing a new subgraph inside. In contrast, the other module consumes the prior subgraph, which is assumed always to take longer than any graph preparation. The quantum bus inside the subcircuit widgets is bilinear, and ancilla qubits are laid out in a comb-like pattern. A third fridge is also connected to subcircuit modules and optimized to host multiple parallelized T-factory widgets up to a limit the `num_intermodule_pipes` parameter allows.

### Version 0.19.3

* Makefile was updated so that repeated executions of `make update-modules` do not emit errors related to existing git directories.

### Version 0.19.2

* Following the notation used by similar libraries and QEC literature, we have switched to the physical-to-logical error scaling relation of P_L=C(p/p_th)^((d+1)/2) for estimation and fitting purposes.

### Version 0.19.1

* Required no. of logical qubits, Delta (scaling all space costs) is calculated significantly more accurately based on Pauli Frames' "path" information. Previously, we used the proxy of max graph degree for Delta, which is not always a valid upper bound.

* A more accurate estimation of Delta means we can now test if RRE's graph stitching was valid to a desired tolerance as done at the end of [test_verify_RRE.py](tests/verify_n_validate/test_verify_RRE.py).

### Version 0.19.0

* A new version of [Jabalizer](https://github.com/QSI-BAQS/Jabalizer.jl/tree/track_paulis), equipped with Pauli Tracking capabilities, was integrated. Jabalizer is now added as a submodule and acts as the default and only available FT compiler. This version allows one to track Pauli Frames and apply corrections. This means for the first time, the graph states can be robustly verified and estimations should be improved.
  * A "robust verification test" was added for common QFT4 graphs in [test_robust_verification_jabalizer.py](tests/verify_n_validate/test_robust_verification_jabalizer.py) unit tests.

*  We have deprecated `.adjlist` format for storing the graph states in favor of new JSON outputs containing both graph and Pauli Frames info. 
