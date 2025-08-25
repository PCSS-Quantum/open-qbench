import time

import dimod
from qlauncher.base import Algorithm, Backend, Problem

from open_qbench.core.benchmark import BaseAnalysis, BenchmarkInput, BenchmarkResult
from open_qbench.core.hybrid_benchmark import HybridBenchmark
from open_qbench.sampler.benchmark_sampler import BenchmarkSampler
from open_qbench.utils import check_tuple_types


class OptimizationBenchmark(HybridBenchmark):
    """
    Benchmark optimization performance of your backend using several analyses.
    """

    def __init__(
        self,
        sampler: dimod.Sampler | tuple[Algorithm, Backend] | BenchmarkSampler,
        benchmark_input: BenchmarkInput,
        analysis: BaseAnalysis,
        name: str = "Optimization Benchmark",
    ) -> None:
        super().__init__(benchmark_input=benchmark_input, analysis=analysis, name=name)
        sampler = (
            BenchmarkSampler(sampler)
            if not isinstance(sampler, BenchmarkSampler)
            else sampler
        )

        self._verify_input(benchmark_input)
        self._verify_sampler(sampler)

        self.input = benchmark_input
        self.sampler = sampler
        self.analysis = analysis

        self.result = BenchmarkResult(self.name, self.benchmark_input)

    def _verify_input(self, benchmark_input: BenchmarkInput) -> None:
        if not isinstance(benchmark_input.program, Problem):
            raise ValueError(
                "This benchmark accepts only optimization problems as input."
            )

    def _verify_sampler(self, sampler: BenchmarkSampler) -> None:
        if not isinstance(sampler.sampler, dimod.Sampler | tuple) or (
            isinstance(sampler.sampler, tuple)
            and not check_tuple_types(sampler.sampler, [Algorithm, Backend])
        ):
            raise ValueError(
                "The provided sampler should be able to solve optimization problems."
                "(use dimod.Sampler or a QLauncher (Algorithm,Backend) tuple)"
            )

    def run(self) -> BenchmarkResult:
        start = time.time()
        self.result.execution_data["counts_backend"] = self.sampler.get_counts(
            self.input.program
        )
        execution_time = time.time() - start
        self.result.metrics["execution_time"] = execution_time

        self.result = self.analysis.run(self.result)
        return self.result
