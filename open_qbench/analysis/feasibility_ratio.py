from collections.abc import Callable

from qlauncher.base import Problem

from open_qbench.core.benchmark import BaseAnalysis, BenchmarkError, BenchmarkResult


class FeasibilityRatioAnalysis(BaseAnalysis):
    """
    Calculates the feasibility ratio (num_feasible / num_total) from the execution results with the provided feasibility metric.

    """

    def __init__(self, feasibility_analysis: Callable[[str, Problem], bool]) -> None:
        super().__init__()
        self.feasibility_analysis = feasibility_analysis

    def run(self, execution_results: BenchmarkResult) -> BenchmarkResult:
        try:
            counts_backend: dict = execution_results.execution_data["dist_backend"]
        except KeyError as e:
            raise BenchmarkError(
                "BenchmarkResult not populated with distributions"
            ) from e

        bench_in = execution_results.input.program
        if not isinstance(bench_in, Problem):
            raise ValueError(
                f"Expected the input program to be an optimization problem, got {type(bench_in)}"
            )

        total_count, feasible_count = 0, 0
        for sample, count in counts_backend.items():
            total_count += count
            if self.feasibility_analysis(sample, bench_in):
                feasible_count += count

        execution_results.metrics["feasibility_ratio"] = feasible_count / total_count
        return execution_results
