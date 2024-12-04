from typing import Sequence, Union, override

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
    def __init__(self, photonic_register: PhotonicRegister):
        super().__init__()
        self.pregs: list[PhotonicRegister] = []
        if not isinstance(photonic_register, PhotonicRegister):
            raise CircuitError("Expected a PhotonicRegister")
        self.pregs.append(photonic_register)
        self._data: list[PhotonicCircuitInstruction] = []

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
            instruction (Operation): _description_
            qargs (Sequence[QumodeSpecifier] | None, optional): _description_. Defaults to None.

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
        self, theta, qumode1, qumode2, label: str | None = None
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
