from typing import override
from ptseries.tbi import create_tbi
from ptseries.tbi.pt1 import PT1AsynchronousResults
import numpy as np

from collections.abc import Iterable

from qiskit.providers import JobError, JobStatus

from qiskit.primitives.base.base_primitive_job import ResultT
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives.containers.primitive_result import PrimitiveResult

from qc_app_benchmarks.photonics import PhotonicCircuitInstruction, PhotonicInstruction, BS, PhotonicCircuit, PhotonicRegister
from qc_app_benchmarks.sampler import BosonicSampler


from qc_app_benchmarks.fidelities import classical_fidelity, normalized_fidelity
from qc_app_benchmarks.fidelity_benchmark import BenchmarkSuite, FidelityBenchmark


class OrcaJob(PrimitiveJob):

    def __init__(self, function, *args, **kwargs):
        super().__init__(function, *args, **kwargs)
        self.orca_results = None
        self.cancelled = lambda: False

    @override
    def _submit(self):
        if self.orca_results is not None:
            raise JobError("Orca job has been submitted already.")
        self.orca_results = self._function(*self._args, **self._kwargs)

    @override
    def result(self) -> ResultT:
        self._check_submitted()
        return self.orca_results.get()

    @override
    def status(self) -> JobStatus:
        self._check_submitted()
        if self.cancelled:
            return JobStatus.CANCELLED
        if self.orca_results.done():
            return JobStatus.DONE
        else:
            return JobStatus.RUNNING

    @override
    def _check_submitted(self):
        if self.orca_results is None:
            raise JobError("Orca Job has not been submitted yet.")

    @override
    def cancel(self):
        self._check_submitted()
        self.cancelled = True
        self.orca_results.cancel()


class OrcaResult(PrimitiveResult):
    def __init__(self, pub_results: list[PT1AsynchronousResults], metadata=None):
        super().__init__(pub_results, metadata)

    def cancel(self):
        for pt1_async_result in self._pub_results:
            pt1_async_result.cancel()

    def get(self) -> PrimitiveResult:
        return PrimitiveResult([pt1_async_result.get() for pt1_async_result in self._pub_results])

    def done(self) -> bool:
        return all([pt1_async_result.is_done for pt1_async_result in self._pub_results])


class OrcaSampler(BosonicSampler):
    """This class is separate from the library as the ptseries SDK
    is not public and we want to avoid adding it as dependency."""

    def __init__(self, default_shots=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_shots = default_shots

    def run(self, pubs: Iterable[tuple[PhotonicCircuit, Iterable[float]]], *, shots: int | None = None, options: dict | None = None):
        if options is None:
            options = {}
        if shots is None:
            shots = self._default_shots
        validated_pubs = [self._extract_lengths(pub) for pub in pubs]
        if "tbi_type" in options and "url" in options and options["tbi_type"] == "PT-1":
            job = OrcaJob(self._run_orca, validated_pubs, options, shots)
        else:
            job = PrimitiveJob(self._run_sim, validated_pubs, options, shots)
        job._submit()
        return job

    def _run_orca(self, validated_pubs, options, shots):
        results = []
        for circuit, thetas, loop_lengths in validated_pubs:
            input_state = circuit.input_state  # TODO Replace with proper input state extraction
            tbi_params = {"tbi_type": options["tbi_type"],
                          "url": options["url"]}
            tbi = create_tbi(loop_lengths=loop_lengths, **tbi_params)
            pt1_async_results = tbi.sample_async(input_state, thetas, shots)
            results.append(pt1_async_results)
        return OrcaResult(results)

    def _run_sim(self, validated_pubs, options, shots):
        results = []
        for circuit, thetas, loop_lengths in validated_pubs:
            input_state = circuit.input_state  # TODO Replace with proper input state extraction
            tbi = create_tbi(loop_lengths=loop_lengths)
            sim_results = tbi.sample(input_state, thetas, shots)
            results.append(sim_results)
        return PrimitiveResult(results)

    def _extract_lengths(self, pub: tuple[PhotonicCircuit, Iterable[float]]) -> tuple[PhotonicCircuit, list[float], list[int]]:
        circuit, thetas = pub
        num_qumods = circuit.pregs[0].size
        instructions: list[PhotonicCircuitInstruction] = circuit._data
        loop_lengths: list[int] = []
        new_thetas: Iterable[float] = []
        current_loop_length: int = 0
        last_position: int = 0
        for instruction, theta in zip(instructions, thetas):
            gate = instruction.operation
            self._BS_validation(theta, gate)
            qumodes = instruction.qumodes
            starting_qumode, ending_qumode = qumodes
            first, second = starting_qumode._index, ending_qumode._index
            assert isinstance(first, int) and isinstance(second, int)
            loop_length = second - first
            if loop_length == current_loop_length and first > last_position:
                assert first - last_position - 1 >= 0
                new_thetas.extend([0] * (first - last_position - 1))
                new_thetas.append(theta)
                last_position = first
            else:
                if len(new_thetas) != 0:
                    assert num_qumods - last_position - current_loop_length - 1 >= 0
                    new_thetas.extend([0] * (num_qumods - last_position - current_loop_length - 1))
                new_thetas.extend([0] * first)
                new_thetas.append(theta)
                loop_lengths.append(loop_length)
                current_loop_length = loop_length
                last_position = first
        return circuit, new_thetas, loop_lengths

    def _BS_validation(self, theta, gate):
        if not isinstance(gate, BS):
            raise TypeError("Orca accepts only BS gates!")
        if theta != gate.params[0]:
            raise TypeError("Conflicting parameters!")


if __name__ == "__main__":
    # Valid Circuits
    ph_circuit1 = PhotonicCircuit(PhotonicRegister(4))
    ph_circuit1.input_state = [1, 1, 1, 1]
    ph_circuit1.bs(np.pi/4, 0, 1)
    ph_circuit1.bs(np.pi/4, 1, 2)
    ph_circuit1.bs(np.pi/4, 2, 3)
    ph_circuit1.bs(np.pi/4, 0, 2)
    ph_circuit1.bs(np.pi/4, 1, 3)
    ph_circuit1.bs(np.pi/4, 0, 3)

    ph_circuit2 = PhotonicCircuit(PhotonicRegister(4))
    ph_circuit2.input_state = [1, 1, 1, 0]
    ph_circuit2.bs(np.pi/4, 0, 1)
    ph_circuit2.bs(np.pi/4, 1, 2)
    ph_circuit2.bs(np.pi/4, 2, 3)
    ph_circuit2.bs(np.pi/4, 0, 1)
    ph_circuit2.bs(np.pi/4, 1, 2)
    ph_circuit2.bs(np.pi/4, 2, 3)

    ideal_sampler = OrcaSampler(default_shots=1024)
    backend_sampler = OrcaSampler(default_shots=1024)

    suite = BenchmarkSuite(
        backend_sampler=backend_sampler,
        ideal_sampler=ideal_sampler,
        calculate_accuracy=normalized_fidelity,
        name="test_suite",
    )

    suite.add_benchmarks(
        [
            [(ph_circuit1, [np.pi/4]*6),],
            [(ph_circuit2, [np.pi/4]*6),]
        ]
    )
    suite.run_all()
    print("Results:")
    for res in suite.results:
        print(
            f"{res.name:>15}: depth = {res.input_properties['normalized_depth']}, fidelity = {res.average_fidelity}"
        )

    suite.save_results("test_res")
