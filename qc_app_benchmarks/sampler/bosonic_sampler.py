from ptseries.tbi.tbi_abstract import TBI

from .base_sampler import BaseBenchmarkSampler, SamplerResult


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


if __name__ == "__main__":
    from ptseries.tbi import TBISingleLoop

    orca_sampler = BosonicSampler(TBISingleLoop(), default_samples=10)
    input_st = (1, 1, 1, 1, 1, 1, 1)
    thetas = [-1.5334, 0.0372, 0.8819, -1.9504, 0.6715, 2.6831]
    print(orca_sampler.run((input_st, thetas)))
