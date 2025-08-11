import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from qiskit import QuantumCircuit
from qiskit.providers import Backend

from open_qbench.photonics import PhotonicCircuit

QuantumProgram = (
    QuantumCircuit
    | tuple[QuantumCircuit, Iterable[float]]
    | PhotonicCircuit
    | tuple[PhotonicCircuit, Iterable[float]]
)
"""A QuantumProgram defines what can be used as BenchmarkInput for executing benchmarks.
"""


class BenchmarkError(Exception):
    """A class for errors raised by benchmarks."""


class BenchmarkInput:
    """An input to a benchmark.

    It can be one of several things:

    * a workflow desribing a complete computational problem,
    * quantum circuit\\*,
    * photonic circuit\\*,
    * QUBO matrix,
    * pulse schedule.

    \\* - currently implemented
    """

    def __init__(
        self,
        program: QuantumProgram,
        backend: Backend | None = None,
    ) -> None:
        # self.program = program
        self.backend = backend

        if isinstance(program, tuple):
            self.program = program[0]
            self.params = program[1]
        else:
            self.program = program
            self.params = None

    def __repr__(self):
        return f"Program: {self.program.name}, Backend: {self.backend}"

    @property
    def width(self):
        if isinstance(self.program, QuantumCircuit):
            return self.program.num_qubits
        if isinstance(self.program, PhotonicCircuit):
            return len(self.program.input_state)


@dataclass
class BenchmarkResult:
    """A dataclass for storing the results of running a benchmark."""

    name: str
    input: BenchmarkInput
    execution_data: dict = field(default_factory=dict)
    metrics: dict[str, int | float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        self_dict = asdict(self)
        self_dict.pop("input")
        self_dict["input"] = {
            "program": self.input.program,
            "params": self.input.params,
        }
        return self_dict

    def save_to_file(self, save_dir: str = "./results"):
        """
        Save result to a json file in the provided directory.

        Args:
            save_dir (str, optional): Folder to save the result to. Defaults to "./results".
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(
            os.path.join(save_dir, self.name + ".json"),
            "w",
            encoding="utf-8",
        ) as file:
            file.write(json.dumps(self.to_dict(), indent=4))


class BaseAnalysis:
    """A class for extracting metrics from benchmark executions."""

    def __init__(self) -> None:
        pass

    def run(self, execution_results: BenchmarkResult) -> BenchmarkResult:
        raise NotImplementedError


class BaseBenchmark(ABC):
    """Abstract class defining the interface of a benchmark.

    A Quantum Benchmark is defined by its input represented by different objects
    depending on the level of the hybrid quantum-classical stack, e.g. by
    quantum circuits and by a protocol, which defines how the benchmark
    is executed and how performance metrics are extracted. This class takes in
    the input and defines the protocol in the `run()` method.

    The method for extracting metrics out of collected `BenchmarkResult`s is defined
    by the `analysis` attribute.
    """

    def __init__(
        self,
        benchmark_input: BenchmarkInput,
        analysis: BaseAnalysis | None,
        name: str | None = None,
    ):
        self.benchmark_input = benchmark_input
        self.analysis = analysis

        if name is not None:
            self.name = name

    def __str__(self) -> str:
        return f"Benchmark {self.name}"

    def __repr__(self) -> str:
        return f"QuantumBenchmark({self.benchmark_input.__repr__()})"

    @abstractmethod
    def run(self) -> BenchmarkResult:
        """Execute the benchmark according to the defined protocol.

        Returns:
            BenchmarkResult: An object containing all the data obtained from benchmark execution.

        """
        raise NotImplementedError
