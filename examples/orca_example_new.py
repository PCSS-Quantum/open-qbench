"""Running this example requires adding your SSH key to https://sdk.orcacomputing.com/ and installing with pip install .[ORCA]"""

from functools import partial

from orca_sampler import OrcaSampler

from open_qbench.application_benchmark import ApplicationBenchmark

# from open_qbench.application_benchmark import BenchmarkSuite
from open_qbench.core import BenchmarkInput
from open_qbench.fidelities import classical_fidelity_orca
from open_qbench.photonics import PhotonicCircuit

ph_circuit1 = PhotonicCircuit.from_tbi_params(
    input_state=[1, 0, 1, 0, 1, 0],
    loop_lengths=[1],
    thetas=[0.8479, -0.0095, 0.2154, -1.3921, 0.0614],
)
fidelity = partial(classical_fidelity_orca, input_state=[1, 0, 1, 0, 1, 0])

ideal_sampler = OrcaSampler(default_shots=1024)
backend_sampler = OrcaSampler(default_shots=1024)

ben_input = BenchmarkInput(ph_circuit1)
orca_ben = ApplicationBenchmark(
    ideal_sampler,
    ideal_sampler,
    ben_input,
    name="test",
    accuracy_measure=fidelity,
)
print(orca_ben.benchmark_input)
orca_ben.run()
print(orca_ben.result)
# normalized_fidelity_orca = create_normalized_fidelity(input_state=[1, 0, 1, 0, 1, 0])

# suite = BenchmarkSuite(
#     backend_sampler=backend_sampler,
#     ideal_sampler=ideal_sampler,
#     calculate_accuracy=normalized_fidelity_orca,
#     name="test_suite",
# )
# suite.add_benchmarks(
#     [
#         [
#             (
#                 ph_circuit1,
#                 [0.8479, -0.0095, 0.2154, -1.3921, 0.0614],
#             ),
#         ],
#     ]
# )
# suite.run_all()
# print("Results:")
# for res in suite.results:
#     print(
#         f"{res.name:>15}: depth = {res.input_properties['normalized_depth']}, fidelity = {res.average_fidelity}"
#     )
# suite.save_results("test_res")

# with open("object_dump", "wb") as f:
#     dill.dump(suite, f)
