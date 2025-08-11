"""
Benchmark manager for running multiple benchmarks together
"""

import asyncio
import inspect
import os
from open_qbench.core.benchmark import BaseBenchmark, BenchmarkResult


async def _as_coro(fn):
    if inspect.iscoroutinefunction(fn):
        return await fn()
    return fn()


class BenchmarkManager:
    def __init__(self, *benchmarks: BaseBenchmark) -> None:
        self.benchmarks: list[BaseBenchmark] = list(benchmarks)
        self.results: list[BenchmarkResult] = []

    def add_benchmarks(self, *benchmarks: BaseBenchmark):
        self.benchmarks += list(benchmarks)

    def run_all(self):
        self.results = [benchmark.run() for benchmark in self.benchmarks]

    async def _run_async_coro(self):
        self.results = await asyncio.gather(*[_as_coro(benchmark.run) for benchmark in self.benchmarks])

    def run_all_async(self):
        asyncio.run(self._run_async_coro())

    def save_results(self, path='./results'):
        os.makedirs(path, exist_ok=True)
        for res in self.results:
            res.save_to_file(path)
