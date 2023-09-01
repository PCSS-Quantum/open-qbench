from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import math
import itertools
from typing import Optional, List, Sequence
import time
import os
import json

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import BaseSampler


class BenchmarkError(Exception):
    """Base class for errors raised by the benchmarking suite"""


@dataclass
class BenchmarkResult:
    """Dataclass for storing the results of running a benchmark"""

    name: str
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


class QuantumBenchmark(ABC):
    """Abstract class defining the interface of a quantum benchmark"""

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend_sampler: BaseSampler,
        ideal_sampler: BaseSampler,
        params: Optional[Sequence] = None,
        name: Optional[str] = None,
    ):
        self.circuit = circuit
        self.backend_sampler = backend_sampler
        self.ideal_sampler = ideal_sampler
        self.params = params
        if name is not None:
            self.name = name
        else:
            self.name = self.circuit.name
        # self.result = BenchmarkResult(name=name)

    def __str__(self) -> str:
        return f"Benchmark {self.name}"

    def __repr__(self) -> str:
        return f"QuantumBenchmark({self.circuit.__repr__()})"

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

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def calculate_fidelity(self, dist_ideal: dict, dist_backend: dict):
        pass

    @abstractmethod
    def measure_creation_time(self):
        pass


class QuantumAppBenchmark(QuantumBenchmark):
    """Application benchmark using the normalized depth and average fidelity
    defined in the document. The circuits are run using the Sampler primitive
    from Qiskit. If not applicable, the run() method can be overriden
    """

    basis_gates = {"rx", "ry", "rz", "cx"}

    def run(self) -> BenchmarkResult:
        job = self.ideal_sampler.run(self.circuit, self.params)
        result = job.result()
        dist_ideal = result.quasi_dists[0].binary_probabilities()

        start = time.time()
        job = self.backend_sampler.run(self.circuit, self.params)
        result = job.result()
        execution_time = time.time() - start
        dist_backend = result.quasi_dists[0].binary_probabilities()

        fidelity = self.calculate_fidelity(dist_ideal, dist_backend)
        return BenchmarkResult(
            self.name, dist_backend, dist_ideal, fidelity, execution_time
        )

    @staticmethod
    def classical_fidelity(dist_a: dict, dist_b: dict) -> float:
        """Compute classical fidelity of two probability distributions

        Args:
            counts_a (dict): Distribution of experiment A
            counts_b (dict): Distribution of experiment B

        Returns:
            float: Classical fidelity
        """
        num_qubits = len(list(dist_a.keys())[0])
        bitstrings = ("".join(i) for i in itertools.product("01", repeat=num_qubits))
        fidelity = 0
        for b in bitstrings:
            p_a = dist_a.get(b, 0)
            p_b = dist_b.get(b, 0)
            fidelity += math.sqrt(p_a * p_b)
        fidelity = fidelity**2
        return fidelity

    def calculate_fidelity(self, dist_ideal: dict, dist_backend: dict) -> float:
        backend_fidelity = self.classical_fidelity(dist_ideal, dist_backend)
        uniform_fidelity = self.classical_fidelity(dist_ideal, self._uniform_dist())

        raw_fidelity = (backend_fidelity - uniform_fidelity) / (1 - uniform_fidelity)

        normalized_fidelity = max([raw_fidelity, 0])
        return normalized_fidelity

    @staticmethod
    def counts_to_dist(counts: dict) -> dict:
        shots = sum(counts.values())
        dist = {x: y / shots for x, y in counts.items()}
        return dist

    def _uniform_dist(self) -> dict:
        num_qubits = self.circuit.num_qubits
        bitstrings = ("".join(i) for i in itertools.product("01", repeat=num_qubits))
        prob = 1 / 2**num_qubits
        dist = {b: prob for b in bitstrings}
        return dist

    def normalized_depth(self) -> int:
        """Returns depth of the circuit after transpiling to the normalized basis gate set.
        By default: {"rx", "ry", "rz", "cx"}

        Returns:
            int: circuit depth
        """
        trans_circuit = transpile(self.circuit, basis_gates=list(self.basis_gates))
        if "measure" in trans_circuit.count_ops().keys():
            return trans_circuit.depth() - 1
        return trans_circuit.depth()

    def measure_creation_time(self):
        pass


# class BenchmarkSuite(Sequence[QuantumBenchmark]):
class BenchmarkSuite(list):
    """Class for aggregating different benchmarks and analysing the results"""

    def __init__(
        self, backend_sampler: BaseSampler, ideal_sampler: BaseSampler, name: str
    ) -> None:
        self.backend_sampler = backend_sampler
        self.ideal_sampler = ideal_sampler
        self.results: List[BenchmarkResult] = []
        self.name: Optional[str] = name

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

    def add_circuits(
        self,
        circuits: Sequence[QuantumCircuit],
        parameters: Sequence[Sequence[float]],
    ):
        if len(circuits) != len(parameters):
            raise BenchmarkError(
                f"The number of circuits {len(circuits)} does not match the number of parameter sequences {len(parameters)}"
            )

        for circ, params in zip(circuits, parameters):
            self.extend(
                [
                    QuantumAppBenchmark(
                        circuit=circ,
                        params=params,
                        backend_sampler=self.backend_sampler,
                        ideal_sampler=self.ideal_sampler,
                        name=circ.name,
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

    def export_qasm(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for ben in self:
            qc = transpile(
                ben.circuit,
                basis_gates=list(QuantumAppBenchmark.basis_gates),
                optimization_level=1,
            )
            if ben.params is not None:
                if isinstance(qc, QuantumCircuit):
                    bounded_qc = qc.bind_parameters(ben.params)
                    bounded_qc.qasm(
                        filename=os.path.join(directory, ben.name + ".qasm")
                    )
                elif isinstance(qc, list):
                    for circ in qc:
                        bounded_circ = circ.bind_parameters(ben.params)
                        bounded_circ.qasm(
                            filename=os.path.join(directory, ben.name + ".qasm")
                        )
