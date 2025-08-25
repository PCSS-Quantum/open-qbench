from abc import abstractmethod

from qiskit import QuantumCircuit, transpile

from open_qbench.core.benchmark import (
    BaseAnalysis,
    BaseBenchmark,
    BenchmarkInput,
    BenchmarkResult,
)
from open_qbench.photonics import PhotonicCircuit


class HighLevelBenchmark(BaseBenchmark):
    def __init__(
        self,
        benchmark_input: BenchmarkInput,
        analysis: BaseAnalysis | None = None,
        name: str = "High Level Benchmark",
        transpiler=None,
    ):
        super().__init__(benchmark_input, analysis, name)
        self.transpiler = transpiler

    def _prepare_input(self):
        if isinstance(self.benchmark_input.program, PhotonicCircuit):
            # TODO: validate circuits for backend
            self.compiled_input = self.benchmark_input.program
        elif isinstance(self.benchmark_input.program, QuantumCircuit):
            self.benchmark_input.program.measure_all()
            self.compiled_input = transpile(
                self.benchmark_input.program, self.benchmark_input.backend
            )
            # TODO: more advanced, customizable transpilation with transpilation barriers
        else:
            self.compiled_input = self.benchmark_input.program

    @abstractmethod
    def run(self) -> BenchmarkResult:
        raise NotImplementedError
