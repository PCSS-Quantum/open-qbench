from typing import Sequence, Union, overload, override

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError

from .photonic_gates import (
    BS,
    PhotonicCircuitInstruction,
    PhotonicGate,
    PhotonicOperation,
    PhotonicRegister,
    Qumode,
)

QumodeSpecifier = Union[
    Qumode,
    PhotonicRegister,
    int,
    slice,
    Sequence[Union[Qumode, int]],
]


class PhotonicCircuit(QuantumCircuit):
    """This class was created to provide a Qiskit-like interface for creating photonic
    quantum circuits.

    Thanks to the PhotonicCircuit type, the :class:'BenchmarkSampler' can recognize
    the type of the circuit and call an appropriate sampler internally, eliminating the
    need to interact with separate samplers for gate-based and photonic quantum computers.
    """

    def __init__(self, photonic_register: PhotonicRegister):
        super().__init__()
        self.pregs: list[PhotonicRegister] = []
        if not isinstance(photonic_register, PhotonicRegister):
            raise CircuitError("Expected a PhotonicRegister")
        self.pregs.append(photonic_register)
        self._data: list[PhotonicCircuitInstruction] = []

    def __init__(self, num_qumodes: int):
        super().__init__()
        self.pregs: list[PhotonicRegister] = []
        self.pregs.append(PhotonicRegister(num_qumodes))
        self._data: list[PhotonicCircuitInstruction] = []

    def __init__(self, input_state: list[int]):
        super().__init__()
        self.pregs: list[PhotonicRegister] = []
        self.pregs.append(PhotonicRegister(len(input_state)))
        self._data: list[PhotonicCircuitInstruction] = []
        self.input_state = input_state

    @override
    def append(self, operation: PhotonicCircuitInstruction, qargs):
        """Perform validation and broadcasting before calling _append"""
        # TODO Implement safe append
        self._check_dups()
        operation.broadcast_arguments()

    def _append(
        self,
        instruction: PhotonicOperation,
        qargs: Sequence[Qumode],
    ) -> PhotonicOperation:
        """Append to circuit directly, without any validation

        Args:
            instruction (PhotonicOperation): The instruction to be appended to the circuit
            qargs (Sequence[Qumode]): Concrete qumodes of the circuit that the operation uses

        Raises:
            CircuitError: If the instruction is not a PhotonicGate

        Returns:
            Operation: The appended instruction
        """
        if not isinstance(instruction, PhotonicGate):
            raise CircuitError("Expected a PhotonicGate")
        circuit_instruction = PhotonicCircuitInstruction(instruction, qargs)
        self._data.append(circuit_instruction)
        return instruction

    def _check_dups(self, qubits: Sequence[Qumode]) -> None:
        """Raise exception if list of qubits contains duplicates."""
        squbits = set(qubits)
        if len(squbits) != len(qubits):
            raise CircuitError("duplicate qubit arguments")

    def bs(
        self,
        theta: float,  # float for now, later extend to Parameter
        qumode1: int | Qumode,
        qumode2: int | Qumode,
        label: str | None = None,
    ) -> PhotonicOperation:
        """Apply BS gate."""
        # this whole thing should go into the safe append()
        if all(isinstance(qm, int) for qm in [qumode1, qumode2]):
            # for now we only consider a singular preg
            qumodes = [self.pregs[0][qumode1], self.pregs[0][qumode2]]
        else:
            # args are already Qumodes
            qumodes = [qumode1, qumode2]
        return self._append(BS(theta, label), qumodes)

    @staticmethod
    def from_tbi_params(input_state: list[int], loop_lengths: list[int], thetas: list[float]):
        thetas_copy = thetas.copy()
        circuit = PhotonicCircuit(input_state=input_state)
        for length in loop_lengths:
            for qumode in range(length, len(input_state)):
                circuit.bs(theta=thetas_copy.pop(0), qumode1=qumode-length, qumode2=qumode)
        return circuit


if __name__ == "__main__":
    input_state = [1, 1, 1, 1]
    loop_lengths = [1, 2, 3]
    thetas = [np.pi/4]*6
    ph_circuit: PhotonicCircuit = PhotonicCircuit.from_tbi_params(input_state,loop_lengths,thetas)
    for i, op in enumerate(ph_circuit):
        assert isinstance(op.operation, BS)
        print(op.operation)
        print(op.qumodes)
        print(op.params)
