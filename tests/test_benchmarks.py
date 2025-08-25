import asyncio
import json
import time
from typing import Any

from dwave.samplers import TabuSampler
from qiskit import QuantumCircuit
from qlauncher.base import Problem

from open_qbench.analysis import FeasibilityRatioAnalysis
from open_qbench.apps.optimization import easy_jssp
from open_qbench.benchmarks import ApplicationBenchmark, OptimizationBenchmark
from open_qbench.core.benchmark import BaseBenchmark, BenchmarkInput, BenchmarkResult
from open_qbench.core.manager import BenchmarkManager
from open_qbench.metrics.feasibilities import JSSPFeasibility
from open_qbench.metrics.fidelities import normalized_fidelity
from open_qbench.photonics.photonic_circuit import PhotonicCircuit
from open_qbench.sampler.benchmark_sampler import BenchmarkSampler


class RetBench(BaseBenchmark):
    def run(self) -> BenchmarkResult:
        return BenchmarkResult(
            f"b{self.benchmark_input.program}",
            self.benchmark_input,
            {},
            {"out": self.benchmark_input.program},
        )


class TestSampler(BenchmarkSampler):
    def __init__(self, output: dict) -> None:
        super().__init__(TabuSampler(), shots=1024)
        self.output = output

    def get_counts(
        self, sampler_input: QuantumCircuit | PhotonicCircuit | Problem
    ) -> dict[Any, int]:
        return self.output


def test_application_benchmark():
    bench = ApplicationBenchmark(
        TestSampler({"0": 51, "1": 49}),
        TestSampler({"0": 50, "1": 50}),
        BenchmarkInput(QuantumCircuit(1)),
        accuracy_measure=normalized_fidelity,
    )

    res = bench.run()
    assert isinstance(res, BenchmarkResult)
    assert res.metrics["fidelity"] == normalized_fidelity(
        {"0": 0.51, "1": 0.49}, {"0": 0.5, "1": 0.5}
    )


def test_optimization_benchmark():
    bench = OptimizationBenchmark(
        TestSampler({"1111": 99, "0101": 1}),
        BenchmarkInput(easy_jssp()),
        analysis=FeasibilityRatioAnalysis(JSSPFeasibility),
    )
    res = bench.run()
    assert isinstance(res, BenchmarkResult)
    assert res.metrics["feasibility_ratio"] == 0.99


def test_add_benchmark():
    benchmarks = []
    bm = BenchmarkManager()
    for i in range(10):
        b = RetBench(BenchmarkInput(i), analysis=None)
        bm.add_benchmarks(b)
        benchmarks.append(b)

    assert bm.benchmarks == benchmarks

    bm = BenchmarkManager(*benchmarks)

    assert bm.benchmarks == benchmarks

    bm = BenchmarkManager()
    bm.add_benchmarks(*benchmarks)

    assert bm.benchmarks == benchmarks


def test_manager_sync():
    bm = BenchmarkManager()
    for i in range(10):
        bm.add_benchmarks(RetBench(BenchmarkInput(i), analysis=None))

    bm.run_all()
    assert [r.metrics["out"] for r in bm.results] == list(range(10))


def test_manager_async():
    bm = BenchmarkManager()
    for i in range(10):
        bm.add_benchmarks(RetBench(BenchmarkInput(i), analysis=None))

    bm.run_all_async()
    assert [r.metrics["out"] for r in bm.results] == list(range(10))


def test_manager_async_time():
    """Check if tasks are executed async"""

    class RetBench(BaseBenchmark):
        async def run(self) -> BenchmarkResult:
            start = time.time()
            await asyncio.sleep(self.benchmark_input.program)
            return BenchmarkResult(
                "b", self.benchmark_input, {}, {"out": time.time() - start}
            )

    bm = BenchmarkManager()
    n = 4
    for i in range(n):
        bm.add_benchmarks(RetBench(BenchmarkInput(i), analysis=None))

    start = time.time()
    bm.run_all_async()
    assert round(time.time() - start, 0) == n - 1
    assert [round(r.metrics["out"], 0) for r in bm.results] == list(range(n))


def test_manager_save(tmp_path):
    bm = BenchmarkManager(
        *[RetBench(BenchmarkInput(i), analysis=None) for i in range(10)]
    )
    bm.run_all()
    bm.save_results(save_dir=tmp_path)

    for i in range(10):
        with open(f"{tmp_path}/b{i}.json", "r+") as f:
            j = json.loads(f.read())
            assert j == bm.results[i].to_dict()
