import asyncio
import time
import numpy as np
from open_qbench.core.manager import BenchmarkManager
from open_qbench.core.benchmark import BaseBenchmark, BenchmarkInput, BenchmarkResult


class RetBench(BaseBenchmark):
    def run(self) -> BenchmarkResult:
        return BenchmarkResult('b', self.benchmark_input, {}, {'out': self.benchmark_input.program})


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
    assert [r.metrics['out'] for r in bm.results] == list(range(10))


def test_manager_async():

    bm = BenchmarkManager()
    for i in range(10):
        bm.add_benchmarks(RetBench(BenchmarkInput(i), analysis=None))

    bm.run_all_async()
    assert [r.metrics['out'] for r in bm.results] == list(range(10))
