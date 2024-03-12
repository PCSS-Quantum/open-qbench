from abc import ABC, abstractmethod
from typing import Union, Sequence

from qiskit.primitives.base.base_sampler import (
    BaseSampler,
    PrimitiveResult,
)
from qiskit import QuantumCircuit

from ptseries.tbi.tbi_abstract import TBI

from dimod import SampleSet, BinaryQuadraticModel
from dimod.core.sampler import Sampler as DwaveSampler


class SamplerResult:
    pass


SamplesLike = Union[dict, SampleSet, PrimitiveResult]


class BaseBenchmarkSampler(ABC):
    def __init__(self, default_samples: int = 1024):
        self.default_samples = default_samples

    @abstractmethod
    def run(self, sampler_input, num_samples) -> SamplerResult:
        pass


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


class BosonicSampler(BaseBenchmarkSampler):
    def __init__(self, sampler: TBI, default_samples: int = 100):
        super().__init__(default_samples=default_samples)
        self.sampler = sampler

    def run(self, sampler_input, num_samples=None) -> SamplerResult:
        input_state = sampler_input[0]
        theta_list = sampler_input[1]
        n_samples = num_samples or self.default_samples
        return self.sampler.sample(
            input_state,
            theta_list,
            n_samples=n_samples,
            output_format="dict",
            n_tiling=1,
        )


class AnnealingSampler(BaseBenchmarkSampler):
    def __init__(self, sampler: DwaveSampler, default_samples: int = 10):
        super().__init__(default_samples=default_samples)
        self.sampler = sampler

    def run(
        self, sampler_input: BinaryQuadraticModel, num_samples: int | None = None
    ) -> SamplerResult:
        return self.sampler.sample(
            bqm=sampler_input, num_reads=num_samples or self.default_samples
        )


if __name__ == "__main__":

    # Qiskit test
    from qiskit_aer.primitives import Sampler
    from qc_app_benchmarks.apps.qaoa import jssp_7q_24d

    qiskit_sampler = CircuitSampler(Sampler())
    print(qiskit_sampler.run(jssp_7q_24d()))

    # Dwave test
    from dimod.reference.samplers import SimulatedAnnealingSampler

    Q = {(0, 0): -1, (0, 1): 1, (1, 2): -4.5}
    bqm = BinaryQuadraticModel.from_qubo(Q)
    dwave_sampler = AnnealingSampler(SimulatedAnnealingSampler(), default_samples=8)
    print(dwave_sampler.run(bqm))

    # Orca test
    from ptseries.tbi import TBISingleLoop

    orca_sampler = BosonicSampler(TBISingleLoop(), default_samples=10)
    input_state = (1, 1, 1, 1, 1, 1, 1)
    thetas = [-1.5334, 0.0372, 0.8819, -1.9504, 0.6715, 2.6831]
    print(orca_sampler.run((input_state, thetas)))
