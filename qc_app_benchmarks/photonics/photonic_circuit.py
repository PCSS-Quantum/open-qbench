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
