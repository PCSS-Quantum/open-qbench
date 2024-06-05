from ptseries.tbi.tbi_abstract import TBI
from collections import defaultdict
import numpy as np

from qc_app_benchmarks.sampler.base_sampler import BaseBenchmarkSampler, SamplerResult


def merge_dicts(dicts):
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)

class BosonicSampler(BaseBenchmarkSampler):
    def __init__(self, sampler: TBI, default_samples: int = 100):
        super().__init__(default_samples=default_samples)
        self.sampler = sampler

    def run(self, sampler_input, num_samples=None) -> SamplerResult:
        input_state = sampler_input[0]
        theta_list = sampler_input[1]
        n_samples = num_samples or self.default_samples
        if n_samples <= 100 or np.sum(input_state) < 3:
            result = self.sampler.sample(
                input_state,
                theta_list,
                n_samples=n_samples,
                output_format="dict"
            )
        else: # >100 is not supported by 3 photons
            result = {}
            num_repetitons = n_samples // 100
            for i in range(num_repetitons):
                partial_result = self.sampler.sample(
                    input_state,
                    theta_list,
                    n_samples=100,
                    output_format="dict"
                )
                result = merge_dicts([result, partial_result])
        partial_result = self.sampler.sample(
            input_state,
            theta_list,
            n_samples=n_samples % 100,
            output_format="dict"
        )
        result = merge_dicts([result, partial_result])

        return {k: (v / n_samples) for (k, v) in result.items()}


if __name__ == "__main__":
    from ptseries.tbi import TBISingleLoop

    orca_sampler = BosonicSampler(TBISingleLoop(), default_samples=10)
    input_st = (1, 1, 1, 1, 1, 1, 1)
    thetas = [-1.5334, 0.0372, 0.8819, -1.9504, 0.6715, 2.6831]
    print(orca_sampler.run((input_st, thetas)))
