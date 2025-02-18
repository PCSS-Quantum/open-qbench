from qc_app_benchmarks.electre import draw_graph
from qc_app_benchmarks.fidelity_benchmark import BenchmarkSuite
from qc_app_benchmarks.sampler.base_sampler import BaseBenchmarkSampler
from qiskit import QuantumCircuit
from pyDecision.algorithm import electre_iii
import numpy as np


def backend_comparison(
    backend_samplers: dict[str, BaseBenchmarkSampler],
    ideal_sampler: BaseBenchmarkSampler,
    fidelity_metric: callable,
    benchmark_inputs: list[QuantumCircuit] | list[tuple[QuantumCircuit, list[float]]] | list[float],
    Q: list[float] = [0.02] * 6,
    P: list[float] = [0.1] * 6,
    V: list[float] = [0.4] * 6,
    W: list[float] = [1] * 6,
    graph_path: str = "graph.png",
):
    if not (len(benchmark_inputs) == len(Q) == len(P) == len(V) == len(W)):
        raise ValueError("benchmark_inputs, Q, P, V, W, should be of the same length")
    suites = [BenchmarkSuite(backend_samplers[name], ideal_sampler, fidelity_metric, name) for name in backend_samplers]
    electre_data = []
    devices = []
    for suite in suites:
        suite.add_benchmarks(benchmark_inputs)
        suite.run_all()
        results = [result.average_fidelity for result in suite.results]
        electre_data.append(results)
        devices.append(suite.name)
    electre_data = np.array(electre_data)
    global_concordance, credibility, rank_D, rank_A, rank_N, rank_P = electre_iii(electre_data, P, Q, V, W, graph=False)
    draw_graph(rank_P, devices, graph_path)
    return global_concordance, credibility, rank_D, rank_A, rank_N, rank_P
