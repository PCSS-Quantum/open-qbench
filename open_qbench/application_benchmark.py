import time
from collections.abc import Callable

from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.primitives import BaseSamplerV2

# from examples.orca_sampler import OrcaSampler
from open_qbench.core import (
    BaseAnalysis,
    BenchmarkError,
    BenchmarkInput,
    BenchmarkResult,
    HighLevelBenchmark,
)
from open_qbench.photonics import PhotonicCircuit

# @dataclass
# class ApplicationBenchmarkResult(BenchmarkResult):
#     """Dataclass for storing the results of running a fidelity benchmark."""

#     input_properties: dict
#     dist_backend: dict
#     dist_ideal: dict

#     def save_to_file(self, path: str = "./results"):
#         for key in list(self.dist_backend.keys()).copy():
#             self.dist_backend["".join(str(x) for x in key)] = self.dist_backend.pop(key)
#         for key in list(self.dist_ideal.keys()).copy():
#             self.dist_ideal["".join(str(x) for x in key)] = self.dist_ideal.pop(key)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         with open(
#             os.path.join(path, self.name + ".json"),
#             "w",
#             encoding="utf-8",
#         ) as file:
#             file.write(json.dumps(asdict(self), indent=4))


class FidelityAnalysis(BaseAnalysis):
    def __init__(self, fidelity_callable: Callable[[dict, dict], float]) -> None:
        self.fidelity_callable = fidelity_callable

    def run(self, execution_results: BenchmarkResult) -> BenchmarkResult:
        try:
            dist_backend: dict = execution_results.execution_data["dist_backend"]
            dist_ideal: dict = execution_results.execution_data["dist_ideal"]
            if isinstance(next(iter(dist_backend.values())), int):
                dist_backend = self.counts_to_probs(dist_backend)
            if isinstance(next(iter(dist_ideal.values())), int):
                dist_ideal = self.counts_to_probs(dist_ideal)
        except KeyError as e:
            raise BenchmarkError(
                "BenchmarkResult not populated with distributions"
            ) from e

        fidelity = self.fidelity_callable(dist_backend, dist_ideal)
        execution_results.metrics["fidelity"] = fidelity

        return execution_results

    @staticmethod
    def counts_to_probs(counts: dict[str, int]) -> dict[str, float]:
        """Convert get_counts() output to probability distributions.

        Args:
            counts (dict[str, int]): _description_

        Returns:
            dict[str, float]: _description_

        """
        # TODO: check if Qiskit provides this.
        return {bits: count / sum(counts.values()) for bits, count in counts.items()}


class ApplicationBenchmark(HighLevelBenchmark):
    """A high-level benchmark, that uses fidelity obtained from comparing two probability distributions as the performance metric."""

    def __init__(
        self,
        backend_sampler,
        reference_state_sampler: BaseSamplerV2,
        benchmark_input: BenchmarkInput,
        name: str | None = None,
        analysis: BaseAnalysis | None = None,
        accuracy_measure: Callable[[dict, dict], float] | None = None,
    ):
        super().__init__(
            benchmark_input,
            analysis,
            name,
        )
        self.backend_sampler = backend_sampler
        self.reference_state_sampler = reference_state_sampler
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

    def run(self):
        """Run the Application Benchmark protocol.

        Returns:
            BenchmarkResult: Probability distributions obtained from execution.

        """
        self._prepare_input()
        # run compiled or logical circuit?
        if isinstance(self.benchmark_input.program, PhotonicCircuit):
            ideal_sampler_counts = self.reference_state_sampler.run(
                [self.compiled_input]
            ).result()[0]
        elif isinstance(self.benchmark_input.program, QuantumCircuit):
            ideal_sampler_counts = (
                self.reference_state_sampler.run([self.compiled_input])
                .result()[0]
                .data.meas.get_counts()
            )

        else:
            raise NotImplementedError

        self.result.execution_data["dist_ideal"] = ideal_sampler_counts

        start = time.time()
        if isinstance(self.benchmark_input.program, PhotonicCircuit):
            backend_sampler_counts = self.backend_sampler.run(
                [self.compiled_input]
            ).result()[0]
        elif isinstance(self.benchmark_input.program, QuantumCircuit):
            backend_sampler_counts = (
                self.backend_sampler.run([self.compiled_input])
                .result()[0]
                .data.meas.get_counts()
            )
            executed_circuit = qasm3.dumps(self.compiled_input)
            self.result.execution_data["width"] = self.benchmark_input.width
            self.result.execution_data["normalized_depth"] = self._normalized_depth(
                self.benchmark_input
            )
            self.result.execution_data["depth_transpiled"] = self.compiled_input.depth()
            self.result.execution_data["executed_circuit"] = executed_circuit
        else:
            raise NotImplementedError

        execution_time = time.time() - start
        self.result.execution_data["dist_backend"] = backend_sampler_counts
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
