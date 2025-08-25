"""Running this example requires adding your SSH key to https://sdk.orcacomputing.com/ and installing with pip install .[ORCA]"""

from open_qbench.analysis import FidelityAnalysis
from open_qbench.apps.circuits.max_cut_orca import max_cut_4_nodes
from open_qbench.benchmarks import ApplicationBenchmark
from open_qbench.core import BenchmarkInput
from open_qbench.metrics.fidelities import classical_fidelity
from open_qbench.orca.sampler import OrcaSampler

ph_circuit1 = max_cut_4_nodes()
ideal_sampler = OrcaSampler(default_shots=1024)
backend_sampler = OrcaSampler(default_shots=1024)

ben_input = BenchmarkInput(ph_circuit1)
orca_ben = ApplicationBenchmark(
    ideal_sampler,
    ideal_sampler,
    ben_input,
    analysis=FidelityAnalysis(classical_fidelity),
    name="test",
)
print(orca_ben.benchmark_input)
orca_ben.run()
print(orca_ben.result)
