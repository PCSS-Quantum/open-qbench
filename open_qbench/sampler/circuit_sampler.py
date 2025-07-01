from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.primitives.base.base_sampler import BaseSamplerV2

from .base_sampler import BaseBenchmarkSampler, SamplerResult


class CircuitSampler(BaseBenchmarkSampler):
    def __init__(self, sampler: BaseSamplerV2, default_samples: int = 1024):
        self.default_samples = default_samples
        self.sampler = sampler

    def _transform_input(sampler_input):
        # if input contains quantum circuits, transpile for backend
        # if photonic circuits - transform for ORCA (only this backend for simplicity, but able to generalize in the future)
        return

    def run(self, sampler_input, num_samples=None) -> SamplerResult:
        if num_samples is None:
            num_samples = self.default_samples
        if isinstance(sampler_input, QuantumCircuit):
            job = self.sampler.run([sampler_input], shots=num_samples)
        elif isinstance(sampler_input, Sequence):
            job = self.sampler.run([sampler_input], shots=num_samples)
        return job.result()[0].data.meas.get_counts()


if __name__ == "__main__":
    from qiskit_aer.primitives import Sampler

    from ..apps.qaoa import jssp_7q_24d

    qiskit_sampler = CircuitSampler(Sampler())
    print(qiskit_sampler.run(jssp_7q_24d()))
