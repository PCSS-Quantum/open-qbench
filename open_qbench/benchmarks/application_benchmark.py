import time
from collections.abc import Callable

import dimod
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.primitives import BaseSamplerV2

from open_qbench.analysis.fidelity import FidelityAnalysis
from open_qbench.core import (
    BaseAnalysis,
    BenchmarkInput,
    BenchmarkResult,
    HighLevelBenchmark,
)
from open_qbench.core.benchmark import BenchmarkError
from open_qbench.sampler.benchmark_sampler import BenchmarkSampler


class ApplicationBenchmark(HighLevelBenchmark):
    """A high-level benchmark, that uses fidelity obtained from comparing two probability distributions as the performance metric."""

    def __init__(
        self,
        backend_sampler: BaseSamplerV2 | dimod.Sampler | BenchmarkSampler,
        reference_state_sampler: BaseSamplerV2 | dimod.Sampler | BenchmarkSampler,
        benchmark_input: BenchmarkInput,
        name: str = "Application Benchmark",
        analysis: BaseAnalysis | None = None,
        accuracy_measure: Callable[[dict, dict], float] | None = None,
    ):
        super().__init__(
            benchmark_input,
            analysis,
            name,
        )
        self.backend_sampler = (
            BenchmarkSampler(backend_sampler)
            if not isinstance(backend_sampler, BenchmarkSampler)
            else backend_sampler
        )
        self.reference_state_sampler = (
            BenchmarkSampler(reference_state_sampler)
            if not isinstance(reference_state_sampler, BenchmarkSampler)
            else reference_state_sampler
        )

        if analysis is not None:
            self.analysis = analysis
        elif accuracy_measure is not None:
            self.analysis = FidelityAnalysis(accuracy_measure)
        else:
            raise BenchmarkError(
                "Analysis has to be defined either directly or by the accuracy_measure argument"
            )
        self.result = BenchmarkResult(self.name, self.benchmark_input)

    basis_gates = frozenset(
        ("rx", "ry", "rz", "cx")
    )  # Gate set used for calculating the normalized circuit depth

    def run(self) -> BenchmarkResult:
        """Run the Application Benchmark protocol.

        Returns:
            BenchmarkResult: Probability distributions obtained from execution.

        """
        self._prepare_input()

        # Not using isinstance() because PhotonicCircuit isinstance of QuantumCircuit and it breaks
        if self.benchmark_input.program.__class__ is QuantumCircuit:
            executed_circuit = qasm3.dumps(self.compiled_input)
            self.result.execution_data["width"] = self.benchmark_input.width
            self.result.execution_data["normalized_depth"] = self._normalized_depth(
                self.benchmark_input
            )
            self.result.execution_data["depth_transpiled"] = self.compiled_input.depth()
            self.result.execution_data["executed_circuit"] = executed_circuit

        self.result.execution_data["counts_ideal"] = (
            self.reference_state_sampler.get_counts(self.compiled_input)
        )

        start = time.time()
        self.result.execution_data["counts_backend"] = self.backend_sampler.get_counts(
            self.compiled_input
        )
        execution_time = time.time() - start

        self.result.metrics["execution_time"] = execution_time

        self.result = self.analysis.run(self.result)

        return self.result

    @staticmethod
    def _normalized_depth(benchmark_input: BenchmarkInput) -> int:
        """Return depth of the circuit after transpiling to the normalized basis gate set.

        Returns:
            int: circuit depth

        """
        if isinstance(benchmark_input.program, QuantumCircuit):
            trans_circuits = transpile(
                benchmark_input.program,
                basis_gates=list(ApplicationBenchmark.basis_gates),
            )
            if "measure" in trans_circuits.count_ops():
                return trans_circuits.depth() - 1
            return trans_circuits.depth()
        else:
            return 0
            # TODO: implement for photonics

    def measure_creation_time(self):
        pass
