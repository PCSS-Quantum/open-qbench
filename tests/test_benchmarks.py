import asyncio
import json
import time

from open_qbench.core.benchmark import BaseBenchmark, BenchmarkInput, BenchmarkResult
from open_qbench.core.manager import BenchmarkManager


class RetBench(BaseBenchmark):
    def run(self) -> BenchmarkResult:
        return BenchmarkResult(
            f"b{self.benchmark_input.program}",
            self.benchmark_input,
            {},
            {"out": self.benchmark_input.program},
        )


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
