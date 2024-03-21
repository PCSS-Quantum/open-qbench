from dataclasses import dataclass, asdict
import time
import os
import json

from qiskit import QuantumCircuit, transpile
from qiskit import qasm3

from .base_benchmark import BaseQuantumBenchmark, BenchmarkResult
from .fidelities import normalized_fidelity
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
        result = self.reference_state_sampler.run(self.benchmark_input)
        dist_ideal = result.binary_probabilities()

        start = time.time()
        result = self.backend_sampler.run(self.benchmark_input)
        execution_time = time.time() - start
        dist_backend = result.binary_probabilities()

        fidelity = self.calculate_accuracy(dist_ideal, dist_backend)

        input_properties = {"normalized_depth": None, "num_q_vars": None}
        return FidelityBenchmarkResult(
            self.name,
            repr(self.benchmark_input),
            input_properties,
            dist_backend,
            dist_ideal,
            fidelity,
            execution_time,
        )

    def calculate_accuracy(self, state_ref: dict, dist_backend: dict):
        return normalized_fidelity(state_ref, dist_backend)

    # def normalized_depth(self) -> int:
    #     """Returns depth of the circuit after transpiling to the normalized basis gate set.
    #     By default: {"rx", "ry", "rz", "cx"}

    #     Returns:
    #         int: circuit depth
    #     """
    #     if isinstance(self.backend_sampler, CircuitSampler):
    #         trans_circuit = transpile(
    #             self.benchmark_input, basis_gates=list(self.basis_gates)
    #         )
    #         if "measure" in trans_circuit.count_ops().keys():
    #             return trans_circuit.depth() - 1
    #         return trans_circuit.depth()
    #     else:
    #         raise NotImplementedError

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
    def backend_sampler(self, sampler_instance: BaseBenchmarkSampler):
        if not isinstance(sampler_instance, BaseBenchmarkSampler):
            raise TypeError(
                "backend_sampler must be an instance of qc_app_benchmarks.sampler.BaseBenchmarkSampler"
            )
        self._backend_sampler = sampler_instance

    @property
    def ideal_sampler(self):
        return self._ideal_sampler

    @ideal_sampler.setter
    def ideal_sampler(self, sampler_instance: BaseBenchmarkSampler):
        if not isinstance(sampler_instance, BaseBenchmarkSampler):
            raise TypeError(
                "ideal_sampler must be an instance of qc_app_benchmarks.sampler.BaseBenchmarkSampler"
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
