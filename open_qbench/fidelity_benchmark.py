import json
import os
import time
from dataclasses import asdict, dataclass
from collections.abc import Callable, Sequence

from qiskit import qasm3, transpile
from qiskit.primitives import BaseSamplerV2

from examples.orca_sampler import OrcaSampler

from .base_benchmark import BaseQuantumBenchmark, BenchmarkResult
from .fidelities import normalized_fidelity
from .sampler import CircuitSampler
from .sampler.base_sampler import BaseBenchmarkSampler


@dataclass
class FidelityBenchmarkResult(BenchmarkResult):
    """Dataclass for storing the results of running a fidelity benchmark"""

    benchmark_input: str
    input_properties: dict
    dist_backend: dict
    dist_ideal: dict
    average_fidelity: float
    execution_time: float

    def save_to_file(self, path: str = "./results"):
        for key in list(self.dist_backend.keys()).copy():
            self.dist_backend["".join(str(x) for x in key)] = self.dist_backend.pop(key)
        for key in list(self.dist_ideal.keys()).copy():
            self.dist_ideal["".join(str(x) for x in key)] = self.dist_ideal.pop(key)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(
            os.path.join(path, self.name + ".json"), "w", encoding="utf-8"
        ) as file:
            file.write(json.dumps(asdict(self), indent=4))


class FidelityBenchmark(BaseQuantumBenchmark):
    """A benchmark which uses a fidelity obtained from comparing two probability
    distributions as its accuracy measure.
    """

    def __init__(self, backend_sampler, reference_state_sampler, benchmark_input, name=None, accuracy_measure: Callable[[dict, dict], float] | None = None):
        super().__init__(backend_sampler, reference_state_sampler, benchmark_input, name)
        self.accuracy_measure = accuracy_measure if accuracy_measure is not None else self.calculate_accuracy
    basis_gates = {"rx", "ry", "rz", "cx"}

    def run(self) -> FidelityBenchmarkResult:
        result = self.reference_state_sampler.run(self.benchmark_input)
        if hasattr(result, "binary_probabilities"):
            dist_ideal = result.binary_probabilities()
        else:
            dist_ideal: dict = result.result()[0]
            dist_ideal = {x: y/sum(dist_ideal.values()) for x, y in dist_ideal.items()}

        start = time.time()
        result = self.backend_sampler.run(self.benchmark_input)
        execution_time = time.time() - start

        if hasattr(result, "binary_probabilities"):
            dist_backend = result.binary_probabilities()
        else:
            dist_backend: dict = result.result()[0]
            dist_backend = {x: y/sum(dist_ideal.values()) for x, y in dist_ideal.items()}

        fidelity = self.accuracy_measure(dist_ideal, dist_backend)

        input_properties = {"normalized_depth": None, "num_q_vars": None}
        if isinstance(self.backend_sampler, CircuitSampler):
            input_properties["normalized_depth"] = self.normalized_depth()
            if isinstance(self.benchmark_input, Sequence):
                name = self.benchmark_input[0].name
            else:
                name = self.benchmark_input.name
        elif isinstance(self.backend_sampler, OrcaSampler):
            name = str(self.benchmark_input[0][0])
        else:
            name = str(self.benchmark_input)
        return FidelityBenchmarkResult(
            name,
            repr(self.benchmark_input),
            input_properties,
            dist_backend,
            dist_ideal,
            fidelity,
            execution_time,
        )

    def calculate_accuracy(self, state_ref: dict, dist_backend: dict):
        return normalized_fidelity(state_ref, dist_backend)

    def normalized_depth(self) -> int:
        """Returns depth of the circuit after transpiling to the normalized basis gate set.
        By default: {"rx", "ry", "rz", "cx"}

        Returns:
            int: circuit depth
        """
        if isinstance(self.backend_sampler, CircuitSampler):
            if isinstance(self.benchmark_input, Sequence):
                circuits = self.benchmark_input[0]
            else:
                circuits = self.benchmark_input
            trans_circuits = transpile(circuits, basis_gates=list(self.basis_gates))
            if "measure" in trans_circuits.count_ops().keys():
                return trans_circuits.depth() - 1
            return trans_circuits.depth()
        else:
            raise NotImplementedError

    def measure_creation_time(self):
        pass


class BenchmarkSuite(list[FidelityBenchmark]):
    """Class for aggregating different benchmarks and analysing the results"""

    def __init__(
        self,
        backend_sampler: BaseBenchmarkSampler,
        ideal_sampler: BaseBenchmarkSampler,
        calculate_accuracy,
        name: str,
    ) -> None:
        self.backend_sampler = backend_sampler
        self.ideal_sampler = ideal_sampler
        self.calculate_accuracy = calculate_accuracy
        self.results: list[FidelityBenchmarkResult] = []
        self.name: str | None = name

    @property
    def backend_sampler(self):
        return self._backend_sampler

    @backend_sampler.setter
    def backend_sampler(self, sampler_instance):
        if not isinstance(sampler_instance, BaseSamplerV2):
            raise TypeError(
                "backend_sampler must be an instance of qiskit.primitives.BaseSamplerV2"
            )
        self._backend_sampler = sampler_instance

    @property
    def ideal_sampler(self):
        return self._ideal_sampler

    @ideal_sampler.setter
    def ideal_sampler(self, sampler_instance):
        if not isinstance(sampler_instance, BaseSamplerV2):
            raise TypeError(
                "ideal_sampler must be an instance of qiskit.primitives.BaseSamplerV2"
            )
        self._ideal_sampler = sampler_instance

    def add_benchmarks(
        self,
        benchmark_inputs,
    ):
        for bench_in in benchmark_inputs:
            self.extend(
                [
                    FidelityBenchmark(
                        backend_sampler=self.backend_sampler,
                        reference_state_sampler=self.ideal_sampler,
                        benchmark_input=bench_in,
                        name=str(bench_in),
                        accuracy_measure=self.calculate_accuracy,
                    )
                ]
            )

    def run_all(self):
        for ben in self:
            result = ben.run()
            self.results.extend([result])

    def save_results(self, directory: str = "./results"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for res in self.results:
            res.save_to_file(directory)

    def plot_results(self):
        pass

    def export_qasm(self, directory: str, *, ver: int = 2):
        if ver not in (2, 3):
            raise ValueError("Only OpenQASM 2.0 and 3.0 are supported")
        if not os.path.exists(directory):
            os.makedirs(directory)
        for ben in self:
            qc = transpile(
                ben.circuit,
                basis_gates=list(FidelityBenchmark.basis_gates),
                optimization_level=1,
            )
        circuits = qc if isinstance(qc, list) else [qc]
        if ben.params is not None:
            for circ in circuits:
                bounded_qc = circ.assign_parameters(ben.params)

                if ver == 2:
                    bounded_qc.qasm(
                        filename=os.path.join(directory, ben.name + ".qasm")
                    )
                elif ver == 3:
                    with open(
                        os.path.join(directory, ben.name + ".qasm3"),
                        "w",
                        encoding="UTF-8",
                    ) as f:
                        qasm3.dump(bounded_qc, f)
            # elif isinstance(qc, list):
            #     for circ in qc:
            #         bounded_circ = circ.assign_parameters(ben.params)
            #         bounded_circ.qasm(
            #             filename=os.path.join(directory, ben.name + ".qasm")
            #         )
