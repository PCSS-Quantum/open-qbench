"""Running this example requires adding your SSH key to https://sdk.orcacomputing.com/ and installing with pip install .[ORCA]"""

from ast import arg
from typing import override
from ptseries.tbi import create_tbi
from ptseries.tbi import PT1
from ptseries.tbi.pt1 import PT1AsynchronousResults
from pprint import pprint
import numpy as np

import uuid
from concurrent.futures import ThreadPoolExecutor

from qiskit.providers import JobError, JobStatus
from qiskit.providers.jobstatus import JOB_FINAL_STATES

from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob, ResultT
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives.containers.primitive_result import PrimitiveResult

from qc_app_benchmarks.photonics import PhotonicCircuitInstruction, PhotonicGate, PhotonicInstruction, BS, PhotonicCircuit, PhotonicRegister
from qc_app_benchmarks.sampler import BosonicSampler


class OrcaJob(PrimitiveJob):

    def __init__(self, function, *args, **kwargs):
        super().__init__(function, *args, **kwargs)
        self.pt_async_res=None
        self.cancelled = False

    @override
    def _submit(self):
        if self._future is not None:
            raise JobError("Primitive job has been submitted already.")

        self.pt_async_res = self._function(*self._args, **self._kwargs)
        
    @override
    def result(self) -> ResultT:
        self._check_submitted()
        return self.pt_async_res.get()
    
    @override
    def status(self) -> JobStatus:
        self._check_submitted()
        if self.cancelled:
            return JobStatus.CANCELLED
        if self.pt_async_res.is_done:
            return JobStatus.DONE
        else:
            return JobStatus.RUNNING
        
    @override
    def _check_submitted(self):
        if self.pt_async_res is None:
            raise JobError("Orca Job has not been submitted yet.")
        
    @override
    def cancel(self):
        self._check_submitted()
        self.cancelled = True
        return self.pt_async_res.cancel()


class OrcaSampler(BosonicSampler):
    """This class is separate from the library as the ptseries SDK
    is not public and we want to avoid adding it as dependency."""

    def run(self, pubs, *, shots=None, **options):
        for circuit, thetas in pubs:
            circuit_length = circuit.pregs[0].size
            input_state = circuit.input_state  # TODO Replace with proper input state extraction
            loop_lengths = self.validate_and_extract_lengths(thetas, circuit, circuit_length)
            if "tbi_type" in options and "url" in options:
                tbi_params = {"tbi_type": options["tbi_type"],
                              "url": options["url"]}
            else:
                tbi_params = {}
            tbi = create_tbi(loop_lengths=loop_lengths, **tbi_params)
            if isinstance(tbi, PT1):
                job = OrcaJob(tbi.sample_async, input_state, thetas, shots)
                job._submit()
            else:
                job = PrimitiveJob(tbi.sample, input_state, thetas, shots)
                job._submit()
            pprint(job.result())

    def validate_and_extract_lengths(self, thetas, circuit, circuit_length):
        instructions: list[PhotonicCircuitInstruction] = circuit._data
        t = 0
        loop_length = None
        loop_lengths = []
        if len(thetas) != len(instructions):
            raise Exception("Number of parameters should be the same as number of gates")
        for instruction, theta in zip(instructions, thetas):
            if loop_length is not None and t+loop_length >= circuit_length:
                if t != 0:
                    loop_lengths.append(loop_length)
                t = 0
            qumodes = instruction.qumodes
            gate: PhotonicInstruction = instruction.operation
            if not isinstance(gate, BS):
                raise TypeError("Orca accepts only BS gates!")
            if theta != gate.params[0]:
                raise Exception("Conflicting parameters!")
            if t == 0:
                loop_length = qumodes[1]._index-qumodes[0]._index
            else:
                if loop_length != qumodes[1]._index-qumodes[0]._index:
                    raise Exception("Qumodes selection not consistent with previous gate!")
            if t+loop_length >= circuit_length:
                continue
            if qumodes[0]._index != t or qumodes[1]._index != t+loop_length:
                raise Exception("Qumodes selection not consistent with previous gate!")
            t += 1
        if t+loop_length < circuit_length:
            raise Exception("Not enough gates")
        else:
            loop_lengths.append(loop_length)
        return loop_lengths


if __name__ == "__main__":
    # Valid Circuit
    ph_circuit = PhotonicCircuit(PhotonicRegister(4))
    ph_circuit.input_state = [1, 1, 1, 1]
    ph_circuit.bs(np.pi/4, 0, 1)
    ph_circuit.bs(np.pi/4, 1, 2)
    ph_circuit.bs(np.pi/4, 2, 3)
    ph_circuit.bs(np.pi/4, 0, 2)
    ph_circuit.bs(np.pi/4, 1, 3)
    ph_circuit.bs(np.pi/4, 0, 3)
    orca_sampler = OrcaSampler().run([(ph_circuit, [np.pi/4]*6)], shots=1000)

    # Invalid Circuit
    ph_circuit = PhotonicCircuit(PhotonicRegister(4))
    ph_circuit.input_state = [1, 1, 1, 1]
    ph_circuit.bs(np.pi/4, 0, 1)
    ph_circuit.bs(np.pi/4, 1, 2)
    ph_circuit.bs(np.pi/4, 0, 2)
    ph_circuit.bs(np.pi/4, 1, 3)
    ph_circuit.bs(np.pi/4, 0, 3)
    # orca_sampler = OrcaSampler().run([(ph_circuit, [np.pi/4]*5)], shots=1000)

    # Invalid Circuit
    ph_circuit = PhotonicCircuit(PhotonicRegister(4))
    ph_circuit.input_state = [1, 1, 1, 1]
    ph_circuit.bs(np.pi/4, 0, 1)
    ph_circuit.bs(np.pi/4, 0, 3)
    ph_circuit.bs(np.pi/4, 2, 3)
    ph_circuit.bs(np.pi/4, 1, 2)
    ph_circuit.bs(np.pi/4, 0, 2)
    ph_circuit.bs(np.pi/4, 1, 3)
    orca_sampler = OrcaSampler().run([(ph_circuit, [np.pi/4]*6)], shots=1000)
