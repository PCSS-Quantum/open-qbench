"""
Benchmark manager for running multiple benchmarks together
"""

import asyncio
import inspect
import os

from open_qbench.core.benchmark import BaseBenchmark, BenchmarkResult


async def _as_coro(fn):
    """Convert function to a coroutine to be ran async"""
    if inspect.iscoroutinefunction(fn):
        return await fn()
    return fn()


class BenchmarkManager:
    """
    A utility class for running multiple benchmarks at once.
    """

    def __init__(self, *benchmarks: BaseBenchmark) -> None:
        self.benchmarks: list[BaseBenchmark] = list(benchmarks)
        self.results: list[BenchmarkResult] = []

    def add_benchmarks(self, *benchmarks: BaseBenchmark):
        """Add benchmarks to execute"""
        self.benchmarks += list(benchmarks)

    def run_all(self):
        """Execute all benchmarks and collect results"""
        self.results = [benchmark.run() for benchmark in self.benchmarks]

    async def _run_async_coro(self):
        self.results = await asyncio.gather(
            *[_as_coro(benchmark.run) for benchmark in self.benchmarks]
        )

    def run_all_async(self):
        """Execute all benchmarks asynchronously and collect results"""
        asyncio.run(self._run_async_coro())

    def save_results(self, save_dir="./results"):
        """
        Save benchmark results to a given path.

        Args:
            save_dir (str, optional): Folder to save the results to. Defaults to "./results".
        """
        os.makedirs(save_dir, exist_ok=True)
        for res in self.results:
            res.save_to_file(save_dir)
