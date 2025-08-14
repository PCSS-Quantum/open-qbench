import time

import dimod
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.primitives import BaseSamplerV2

from open_qbench.core import (
    BaseAnalysis,
    BenchmarkInput,
    BenchmarkResult,
    HighLevelBenchmark,
)
from open_qbench.sampler.benchmark_sampler import BenchmarkSampler


class ApplicationBenchmark(HighLevelBenchmark):
    """A high-level benchmark, that uses fidelity obtained from comparing two probability distributions as the performance metric."""

    def __init__(
        self,
        benchmark_input: BenchmarkInput,
        backend_sampler: BaseSamplerV2 | dimod.Sampler | BenchmarkSampler,
        analysis: BaseAnalysis,
        reference_state_sampler: BaseSamplerV2 | dimod.Sampler | None = None,
        name: str | None = None,
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
        if reference_state_sampler is None:
            self.reference_state_sampler = None
        else:
            self.reference_state_sampler = (
                BenchmarkSampler(reference_state_sampler)
                if not isinstance(reference_state_sampler, BenchmarkSampler)
                else reference_state_sampler
            )

        self.analysis = analysis
        self.result = BenchmarkResult(self.name, self.benchmark_input)

    basis_gates = frozenset(
        ("rx", "ry", "rz", "cx")
    )  # Gate set used for calculating the normalized circuit depth

    def run(self):
        """Run the Application Benchmark protocol.

        Returns:
            BenchmarkResult: Probability distributions obtained from execution.

        """
        self._prepare_input()
        # run compiled or logical circuit?
        if self.reference_state_sampler is not None:
            self.result.execution_data["dist_ideal"] = (
                self.reference_state_sampler.get_counts(self.compiled_input)
            )

        # Not using isinstance() because PhotonicCircuit isinstance of QuantumCircuit and it breaks
        if self.benchmark_input.program.__class__ is QuantumCircuit:
            executed_circuit = qasm3.dumps(self.compiled_input)
            self.result.execution_data["width"] = self.benchmark_input.width
            self.result.execution_data["normalized_depth"] = self._normalized_depth(
                self.benchmark_input
            )
            self.result.execution_data["depth_transpiled"] = self.compiled_input.depth()
            self.result.execution_data["executed_circuit"] = executed_circuit

        start = time.time()
        self.result.execution_data["dist_backend"] = self.backend_sampler.get_counts(
            self.compiled_input
        )
        execution_time = time.time() - start
        self.result.metrics["execution_time"] = execution_time

        self.result = self.analysis.run(self.result)

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
