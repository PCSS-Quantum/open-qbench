from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import Sampler
from qiskit_ibm_runtime.fake_provider import FakeGeneva

from open_qbench import ApplicationBenchmark
from open_qbench.apps import grover
from open_qbench.core import BenchmarkInput
from open_qbench.fidelities import normalized_fidelity

ideal_sampler = AerSampler(default_shots=1000)
backend_sampler = Sampler(FakeGeneva())

ab = ApplicationBenchmark(
    backend_sampler,
    ideal_sampler,
    BenchmarkInput(grover.grover_nq(3, 6), backend_sampler.backend()),
    name="Grover_benchmark",
    accuracy_measure=normalized_fidelity,
)

ab.run()
print(ab.result)
