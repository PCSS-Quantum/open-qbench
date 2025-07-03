from collections.abc import Sequence
from qiskit.circuit import ParameterExpression, CircuitError
from qiskit import QuantumCircuit
import numpy as np

from open_qbench.photonics.photonic_gates import (
    BS,
    PhotonicCircuitInstruction,
    PhotonicGate,
    PhotonicOperation,
    PhotonicRegister,
    Qumode,
)


PRINTING_ENABLED: bool = True
try:
    from ptseries.tbi.representation.representation import Drawer
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    PRINTING_ENABLED = False

QumodeSpecifier = Qumode | PhotonicRegister | int | slice | Sequence[Qumode | int]


class PhotonicCircuit(QuantumCircuit):
    """This class was created to provide a Qiskit-like interface for creating photonic
    quantum circuits.

    Thanks to the PhotonicCircuit type, the :class:'BenchmarkSampler' can recognize
    the type of the circuit and call an appropriate sampler internally, eliminating the
    need to interact with separate samplers for gate-based and photonic quantum computers.
    """

    def __init__(self, *regs: PhotonicRegister | int, input_state: list[int] | None = None):
        super().__init__()
        self.pregs: list[PhotonicRegister] = []
        self._data: list[PhotonicCircuitInstruction] = []
        self.input_state: list[int] = input_state if input_state is not None else []
        for reg in regs:
            if isinstance(reg, PhotonicRegister):
                self.pregs.append(reg)
            elif isinstance(reg, int):
                self.pregs.append(PhotonicRegister(reg))
            else:
                raise ValueError(f'Wrong Argument passed as an register to {self.__class__.__name__}')
        if len(regs) == 0 and input_state is not None:
            self.pregs.append(PhotonicRegister(len(input_state)))

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
        theta: ParameterExpression,  # float for now, later extend to Parameter
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

    def draw(self, padding: int = 1, draw: bool = True):
        """Draw function for Photonic Circuits, currently only Orca circuits supported (because of loop lengths).

        Args:
            padding (int, optional): Padding. Defaults to 1.

        Raises:
            ModuleNotFoundError: If optional dependencies are not installed properly this exception is raised.
        """
        if PRINTING_ENABLED is False:
            raise ModuleNotFoundError('To use `PhotonicCircuit.draw` method you need to install `[ORCA]` optional dependencies')
        # loop_length_calculation
        loop_length = 0
        loop_lengths: list[int] = []
        current_loop_length, last_position = 0, 0
        for instruction in self._data:
            qumodes = instruction.qumodes
            starting_qumode, ending_qumode = qumodes
            first, second = starting_qumode._index, ending_qumode._index
            loop_length = second - first
            if loop_length == current_loop_length and first > last_position:
                continue
            loop_lengths.append(loop_length)
            current_loop_length = loop_length
            last_position = first
        input_state = self.input_state if self.input_state else [0] * int(sum(len(preg) for preg in self.pregs))
        n_modes = len(input_state)
        representation = Drawer()  # type: ignore
        structure = representation.get_structure(n_modes, len(loop_lengths), loop_lengths)
        if draw:
            representation.draw(structure, input_state, padding=padding)
            plt.show()  # type: ignore

    @staticmethod
    def from_tbi_params(input_state: list[int], loop_lengths: list[int], thetas: list[float]):
        thetas_copy = thetas.copy()
        circuit = PhotonicCircuit(input_state=input_state)
        for length in loop_lengths:
            for qumode in range(length, len(input_state)):
                circuit.bs(theta=thetas_copy.pop(0), qumode1=qumode-length, qumode2=qumode)
        return circuit

    def __str__(self):
        return self.__class__.__name__ + "_"+''.join(str(x) for x in self.input_state)


if __name__ == "__main__":
    input_state = [1, 1, 1, 1]
    loop_lengths = [1, 2, 3]
    expected_qumodes = []
    for length in loop_lengths:
        for qumode in range(length, len(input_state)):
            expected_qumodes.append((qumode-length, qumode))
    thetas = [np.pi/4]*6
    ph_circuit: PhotonicCircuit = PhotonicCircuit.from_tbi_params(input_state, loop_lengths, thetas)
    for i, op in enumerate(ph_circuit):
        assert isinstance(op.operation, BS)
        assert isinstance(op.operation, PhotonicGate)
        assert op.qumodes[0]._index == expected_qumodes[i][0]
        assert op.qumodes[1]._index == expected_qumodes[i][1]
        assert op.params[0] == thetas[i]
