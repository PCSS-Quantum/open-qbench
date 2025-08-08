"""Running this example requires adding your SSH key to https://sdk.orcacomputing.com/ and installing with pip install .[ORCA]"""

from functools import partial

from orca_sampler import OrcaSampler

from open_qbench.application_benchmark import ApplicationBenchmark
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
