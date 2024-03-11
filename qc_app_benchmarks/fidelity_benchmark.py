from dataclasses import dataclass, asdict
from typing import Sequence
import time
import os
import json

from qiskit import QuantumCircuit, transpile
from qiskit import qasm3
from qiskit.primitives import BaseSampler

from .base_benchmark import BaseQuantumBenchmark
from .fidelities import normalized_fidelity


class BenchmarkError(Exception):
    """Base class for errors raised by the benchmarking suite"""


@dataclass
class FidelityBenchmarkResult:
    """Dataclass for storing the results of running a fidelity benchmark"""

    name: str
    num_qubits: int
    normalized_depth: int
    dist_backend: dict
    dist_ideal: dict
    average_fidelity: float
    execution_time: float

    def save_to_file(self, path: str = "./results"):
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

    basis_gates = {"rx", "ry", "rz", "cx"}

    def run(self) -> FidelityBenchmarkResult:
        if isinstance(self.benchmark_input, Sequence):
            job = self.reference_state_sampler.run(
                self.benchmark_input[0], self.benchmark_input[1]
            )
        else:
            job = self.reference_state_sampler.run(self.benchmark_input)
        result = job.result()
        dist_ideal = result.quasi_dists[0].binary_probabilities()

        start = time.time()
        if isinstance(self.benchmark_input, Sequence):
            job = self.backend_sampler.run(
                self.benchmark_input[0], self.benchmark_input[1]
            )
        else:
            job = self.backend_sampler.run(self.benchmark_input)
        result = job.result()
        execution_time = time.time() - start
        dist_backend = result.quasi_dists[0].binary_probabilities()

        fidelity = self.calculate_accuracy(dist_ideal, dist_backend)

        return FidelityBenchmarkResult(
            self.name,
            self.benchmark_input,
            # self.normalized_depth(),
            0,  # TO DO: fix normalized_depth
            dist_backend,
            dist_ideal,
            fidelity,
            execution_time,
        )

    def calculate_accuracy(self, dist_ref: dict, dist_backend: dict):
        return normalized_fidelity(dist_ref, dist_backend)

    def normalized_depth(self) -> int:
        """Returns depth of the circuit after transpiling to the normalized basis gate set.
        By default: {"rx", "ry", "rz", "cx"}

        Returns:
            int: circuit depth
        """
        if isinstance(self.benchmark_input, QuantumCircuit):
            trans_circuit = transpile(
                self.benchmark_input, basis_gates=list(self.basis_gates)
            )
            if "measure" in trans_circuit.count_ops().keys():
                return trans_circuit.depth() - 1
            return trans_circuit.depth()
        else:
            raise NotImplementedError

    def measure_creation_time(self):
        pass


class BenchmarkSuite(list[FidelityBenchmark]):
    """Class for aggregating different benchmarks and analysing the results"""

    def __init__(
        self,
        backend_sampler: BaseSampler,
        ideal_sampler: BaseSampler,
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
    def backend_sampler(self, sampler_instance: BaseSampler):
        if not isinstance(sampler_instance, BaseSampler):
            raise TypeError(
                "backend_sampler must be an instance of qiskit.primitives.BaseSampler"
            )
        self._backend_sampler = sampler_instance

    @property
    def ideal_sampler(self):
        return self._ideal_sampler

    @ideal_sampler.setter
    def ideal_sampler(self, sampler_instance: BaseSampler):
        if not isinstance(sampler_instance, BaseSampler):
            raise TypeError(
                "ideal_sampler must be an instance of qiskit.primitives.BaseSampler"
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
            if ben.params is not None:
                if isinstance(qc, QuantumCircuit):
                    bounded_qc = qc.assign_parameters(ben.params)

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
                elif isinstance(qc, list):
                    for circ in qc:
                        bounded_circ = circ.assign_parameters(ben.params)
                        bounded_circ.qasm(
                            filename=os.path.join(directory, ben.name + ".qasm")
                        )
