from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.primitives.base.base_sampler import BaseSampler

from .base_sampler import BaseBenchmarkSampler, SamplerResult


class CircuitSampler(BaseBenchmarkSampler):
    def __init__(self, sampler: BaseSampler, default_samples: int = 1024):
        super().__init__(default_samples=default_samples)
        self.sampler = sampler

    def run(self, sampler_input, num_samples=None) -> SamplerResult:
        if isinstance(sampler_input, QuantumCircuit):
            job = self.sampler.run(sampler_input, shots=num_samples)
        elif isinstance(sampler_input, Sequence) and len(sampler_input) == 2:
            circuits = sampler_input[0]
            parameter_values = sampler_input[1]
            job = self.sampler.run(circuits, parameter_values, shots=num_samples)
        return job.result().quasi_dists[0]


if __name__ == "__main__":
    from qiskit_aer.primitives import Sampler
    from ..apps.qaoa import jssp_7q_24d

    qiskit_sampler = CircuitSampler(Sampler())
    print(qiskit_sampler.run(jssp_7q_24d()))
