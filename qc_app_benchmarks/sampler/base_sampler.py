from abc import ABC, abstractmethod


class SamplerResult:
    pass


class BaseBenchmarkSampler(ABC):
    def __init__(self, default_samples: int = 1024):
        self.default_samples = default_samples

    @abstractmethod
    def run(self, sampler_input, num_samples) -> SamplerResult:
        pass
