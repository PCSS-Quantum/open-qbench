from abc import abstractmethod

from .benchmark import BaseBenchmark, BenchmarkResult


class HybridBenchmark(BaseBenchmark):
    def __init__(self):
        pass

    @abstractmethod
    def run(self) -> BenchmarkResult:
        raise NotImplementedError
