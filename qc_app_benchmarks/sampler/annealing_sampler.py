from dimod import BinaryQuadraticModel
from dimod.core.sampler import Sampler as DwaveSampler

from .base_sampler import BaseBenchmarkSampler, SamplerResult


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
    from dimod.reference.samplers import SimulatedAnnealingSampler

    Q = {(0, 0): -1, (0, 1): 1, (1, 2): -4.5}
    bqm = BinaryQuadraticModel.from_qubo(Q)
    dwave_sampler = AnnealingSampler(SimulatedAnnealingSampler(), default_samples=8)
    print(dwave_sampler.run(bqm))
