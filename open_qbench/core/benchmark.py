import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any

from qiskit import QuantumCircuit
from qiskit.providers import Backend

from open_qbench.photonics import PhotonicCircuit

try:
    from qlauncher.base import Problem as Problem
except ImportError:

    class Problem:  # type: ignore[no-redef]
        pass


type QuantumProgram = (
    QuantumCircuit
    | tuple[QuantumCircuit, Iterable[float]]
    | PhotonicCircuit
    | tuple[PhotonicCircuit, Iterable[float]]
    | Problem
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
    * QUBO matrix\\*,
    * pulse schedule.

    \\* - currently implemented
    """

    def __init__(
        self,
        program: QuantumProgram,
        backend: Backend | None = None,
        options: dict | None = None,
    ) -> None:
        self.backend = backend
        if options is not None:
            self.options = options
        else:
            options = {}

        if isinstance(program, tuple):
            self.program = program[0]
            self.params = program[1]
        else:
            self.program = program
            self.params = None

    def __str__(self):
        return f"Program: {self.program.name}, Backend: {type(self.backend).__module__}.{type(self.backend).__qualname__}, Options: {self.options}"

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
    timestamp: datetime = field(default_factory=datetime.now)

    def show(
        self, *fields: str, verbose: bool = False, save_to: Path | str | None = None
    ) -> None:
        """
        A convenience method for displaying, formatting and saving results to files."
        """
        all_output: dict = {
            "name": self.name,
            "input": str(self.input),
            "execution_data": self.execution_data,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(timespec="milliseconds"),
        }

        if save_to is not None:
            if fields:
                output = {
                    **{k: v for k, v in all_output.items() if k != "execution_data"},
                    "execution_data": {
                        k: v
                        for k, v in all_output["execution_data"].items()
                        if k in fields
                    },
                }
            else:
                output = all_output
            path = Path(save_to)
            if path.suffix == "":
                path.mkdir(parents=True, exist_ok=True)
                timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
                path = path / f"{timestamp_str}-{self.name}.json"
            path.write_text(json.dumps(output, indent=4))
        else:
            if verbose:
                output = all_output
            elif fields:
                output = {
                    **{k: v for k, v in all_output.items() if k != "execution_data"},
                    "execution_data": {
                        k: v
                        for k, v in all_output["execution_data"].items()
                        if k in fields
                    },
                }
            else:
                output = {k: v for k, v in all_output.items() if k != "execution_data"}
            pprint(output, sort_dicts=False)

    def __str__(self) -> str:
        execution_data = {
            k: v for k, v in self.execution_data.items() if k != "executed_circuit"
        }
        return (
            f"BenchmarkResult\n"
            f"  name:           {self.name}\n"
            f"  input:          {self.input}\n"
            f"  execution_data: {execution_data}\n"
            f"  metrics:        {self.metrics}\n"
            f"  timestamp:      {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    def to_dict(self) -> dict[str, Any]:
        self_dict = asdict(self)
        self_dict.pop("input")
        self_dict["input"] = {
            "program": self.input.program,
            "params": self.input.params,
        }
        return self_dict


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
        name: str = "Base Benchmark",
    ):
        self.benchmark_input = benchmark_input
        self.analysis = analysis

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
