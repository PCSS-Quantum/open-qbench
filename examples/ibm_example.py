from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import Sampler
from qiskit_ibm_runtime.fake_provider import FakeGeneva

from open_qbench import ApplicationBenchmark
from open_qbench.analysis import FidelityAnalysis
from open_qbench.apps.circuits import grover
from open_qbench.core import BenchmarkInput
from open_qbench.metrics.fidelities import normalized_fidelity

ideal_sampler = AerSampler(default_shots=1000)
backend_sampler = Sampler(FakeGeneva())

ab = ApplicationBenchmark(
    BenchmarkInput(grover.grover_nq(3, 6), backend_sampler.backend()),
    backend_sampler,
    analysis=FidelityAnalysis(normalized_fidelity),
    reference_state_sampler=ideal_sampler,
    name="Grover_benchmark",
)

ab.run()
print(ab.result)
