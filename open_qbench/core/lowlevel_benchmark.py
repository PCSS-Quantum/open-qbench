from abc import abstractmethod

from .benchmark import BaseBenchmark, BenchmarkResult


class LowLevelBenchmark(BaseBenchmark):
    def __init__(self):
        pass

    @abstractmethod
    def run(self) -> BenchmarkResult:
        raise NotImplementedError
