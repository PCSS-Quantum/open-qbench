from collections.abc import Callable

from open_qbench.core.benchmark import BaseAnalysis, BenchmarkError, BenchmarkResult


class FidelityAnalysis(BaseAnalysis):
    def __init__(self, fidelity_callable: Callable[[dict, dict], float]) -> None:
        self.fidelity_callable = fidelity_callable

    def run(self, execution_results: BenchmarkResult) -> BenchmarkResult:
        try:
            dist_backend: dict = execution_results.execution_data["dist_backend"]
            dist_ideal: dict = execution_results.execution_data["dist_ideal"]
            if isinstance(next(iter(dist_backend.values())), int):
                dist_backend = self.counts_to_probs(dist_backend)
            if isinstance(next(iter(dist_ideal.values())), int):
                dist_ideal = self.counts_to_probs(dist_ideal)
        except KeyError as e:
            raise BenchmarkError(
                "BenchmarkResult not populated with distributions"
            ) from e

        fidelity = self.fidelity_callable(dist_backend, dist_ideal)
        execution_results.metrics["fidelity"] = fidelity

        return execution_results

    @staticmethod
    def counts_to_probs(counts: dict[str, int]) -> dict[str, float]:
        """Convert get_counts() output to probability distributions.

        Args:
            counts (dict[str, int]): _description_

        Returns:
            dict[str, float]: _description_

        """
        sum_vals = sum(counts.values())
        return {bits: count / sum_vals for bits, count in counts.items()}
