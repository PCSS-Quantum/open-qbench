import numpy as np
from pyDecision.algorithm import electre_iii

from open_qbench.core.benchmark import BenchmarkResult

from .utils import draw_graph


class ElectreBackendComparison:
    """
    Compare benchmark results with Electre III
    """

    def __init__(
        self,
        metrics: list[tuple[str, float, float, float, float]] | None = None,
        graph_path: str | None = None,
    ) -> None:
        """
        Args:
            metrics (list[tuple[str,float,float,float,float]] | None, optional):
                Metrics to include in the comparison. Each metric should be in the format (name, indifference_threshold, preference_threshold, veto_threshold, weight).
                If None defaults to [('fidelity', 0.02, 0.1, 0.4, 1.0)]. Defaults to None.
            graph_path (str | None, optional): Path for the saved outranking graph, if None no graph is generated. Defaults to None.

        Raises:
            ValueError: If metrics is an empty list.
        """
        if metrics is None:
            metrics = [("fidelity", 0.02, 0.1, 0.4, 1.0)]

        if len(metrics) == 0:
            raise ValueError("Provide at least one metric!")

        self.metrics = []
        self.q = []
        self.p = []
        self.v = []
        self.w = []

        for met, q, p, v, w in metrics:
            self.metrics.append(met)
            self.q.append(q)
            self.p.append(p)
            self.v.append(v)
            self.w.append(w)

        self.graph_path = graph_path

    def run(
        self, *execution_results: BenchmarkResult
    ) -> tuple[np.ndarray, np.ndarray, list, list, list, np.ndarray]:
        electre_data = [
            [res.metrics[m] for m in self.metrics] for res in execution_results
        ]

        electre_data = np.array(electre_data)
        global_concordance, credibility, rank_D, rank_A, rank_N, rank_P = electre_iii(
            electre_data, self.p, self.q, self.v, self.w, graph=False
        )

        if self.graph_path is not None:
            devices = [
                res.backend_name if res.backend_name is not None else res.name
                for res in execution_results
            ]
            draw_graph(rank_P, devices, self.graph_path)

        return global_concordance, credibility, rank_D, rank_A, rank_N, rank_P
