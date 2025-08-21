from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitError
from qiskit.circuit.quantumregister import Qubit

from open_qbench.photonics.photonic_gates import (
    BS,
    PhotonicCircuitInstruction,
    PhotonicGate,
    PhotonicInstruction,
    PhotonicOperation,
    PhotonicRegister,
    Qumode,
)

PRINTING_ENABLED: bool = True
try:
    import matplotlib.pyplot as plt
    from ptseries.tbi.representation.representation import Drawer
except ModuleNotFoundError:
    PRINTING_ENABLED = False

type QumodeSpecifier = Qumode | PhotonicRegister | int | slice | Sequence[Qumode | int]


class PhotonicCircuit(QuantumCircuit):
    """A class created to provide a Qiskit-like interface for creating photonic quantum circuits.

    Thanks to the PhotonicCircuit type, the :class:'BenchmarkSampler' can recognize
    the type of the circuit and call an appropriate sampler internally, eliminating the
    need to interact with separate samplers for gate-based and photonic quantum computers.
    """

    def __init__(
        self,
        *regs: PhotonicRegister | int,
        input_state: list[int] | None = None,
    ):
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
                raise ValueError(
                    f"Wrong Argument passed as an register to {self.__class__.__name__}"
                )
        if len(regs) == 0 and input_state is not None:
            self.pregs.append(PhotonicRegister(len(input_state)))

    def append(
        self, instruction: PhotonicInstruction, qargs: list[QumodeSpecifier]
    ) -> PhotonicOperation | list[PhotonicOperation]:
        """Perform validation and broadcasting before calling _append."""
        if len(qargs) == 1:
            combs = [(x,) for x in self._get_qumodes(qargs[0])]
        elif len(qargs) == 2:
            left, right = self._get_qumodes(qargs[0]), self._get_qumodes(qargs[1])
            combs = self._broadcast_qumodes(left, right)
        else:
            raise ValueError(
                "Only operations on one or two qumodes are supported right now."
            )

        ops: list[PhotonicOperation] = []
        for comb in combs:
            self._check_dups(comb)
            ops.append(self._append(instruction, comb))
        return ops[0] if len(ops) == 1 else ops

    def _append(
        self,
        instruction: PhotonicOperation,
        qargs: Sequence[Qumode],
    ) -> PhotonicOperation:
        """Append to circuit directly, without any validation.

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

    def _get_qumodes(self, qumode_specifier: QumodeSpecifier) -> list[Qumode]:
        if isinstance(qumode_specifier, Qumode):
            return [qumode_specifier]
        elif isinstance(qumode_specifier, PhotonicRegister):
            return [x for x in qumode_specifier]
        elif isinstance(qumode_specifier, int):
            return [self.pregs[0][qumode_specifier]]
        elif isinstance(qumode_specifier, slice):
            return self.pregs[0][qumode_specifier]
        elif isinstance(qumode_specifier, Sequence):
            qumodes = [self._get_qumodes(qm) for qm in qumode_specifier]
            ret = []
            for qm in qumodes:
                ret += qm
            return ret

    def _broadcast_qumodes(
        self, left: list[Qumode], right: list[Qumode]
    ) -> list[tuple[Qumode, Qumode]]:
        if len(left) == len(right):
            return list(zip(left, right, strict=False))
        elif len(left) == 1:
            return [(left[0], right_qm) for right_qm in right]
        elif len(right) == 1:
            return [(left_qm, right[0]) for left_qm in left]
        else:
            raise CircuitError(
                f"Not sure how to broadcast these qumodes {[left, right]}"
            )

    def bs(
        self,
        theta: float,  # float for now, later extend to Parameter
        qumode1: QumodeSpecifier,
        qumode2: QumodeSpecifier,
        label: str | None = None,
    ) -> PhotonicOperation | list[PhotonicOperation]:
        """Apply BS gate."""
        return self.append(BS(theta, label), [qumode1, qumode2])

    @property
    def qubits(self) -> list[Qubit]:
        raise CircuitError("This circuit does not have qubits.")

    def draw(self, padding: int = 1, draw: bool = True):
        """Draw function for Photonic Circuits, currently only Orca circuits supported (because of loop lengths).

        Args:
            padding (int, optional): Padding. Defaults to 1.

        Raises:
            ModuleNotFoundError: If optional dependencies are not installed properly this exception is raised.

        """
        if PRINTING_ENABLED is False:
            raise ModuleNotFoundError(
                "To use `PhotonicCircuit.draw` method you need to install `[ORCA]` optional dependencies"
            )
        # loop_length_calculation
        loop_length = 0
        loop_lengths: list[int] = []
        current_loop_length, last_position = 0, 0
        for instruction in self._data:
            qumodes = instruction.qumodes
            starting_qumode, ending_qumode = qumodes
            first, second = (
                starting_qumode._index,
                ending_qumode._index,
            )
            loop_length = second - first
            if loop_length == current_loop_length and first > last_position:
                continue
            loop_lengths.append(loop_length)
            current_loop_length = loop_length
            last_position = first
        input_state = (
            self.input_state
            if self.input_state
            else [0] * int(sum(len(preg) for preg in self.pregs))
        )
        n_modes = len(input_state)
        representation = Drawer()  # type: ignore
        structure = representation.get_structure(
            n_modes, len(loop_lengths), loop_lengths
        )
        if draw:
            representation.draw(structure, input_state, padding=padding)
            plt.show()  # type: ignore

    @classmethod
    def from_tbi_params(
        cls,
        input_state: list[int],
        loop_lengths: list[int],
        thetas: list[float],
    ):
        thetas_copy = thetas.copy()
        circuit = PhotonicCircuit(input_state=input_state)
        for length in loop_lengths:
            for qumode in range(length, len(input_state)):
                circuit.bs(
                    theta=thetas_copy.pop(0),
                    qumode1=qumode - length,
                    qumode2=qumode,
                )
        return circuit

    def __str__(self):
        return self.__class__.__name__ + "_" + "".join(str(x) for x in self.input_state)


if __name__ == "__main__":
    input_state = [1, 1, 1, 1]
    loop_lengths = [1, 2, 3]
    expected_qumodes = []
    for length in loop_lengths:
        for qumode in range(length, len(input_state)):
            expected_qumodes.append((qumode - length, qumode))
    thetas = [np.pi / 4] * 6
    ph_circuit: PhotonicCircuit = PhotonicCircuit.from_tbi_params(
        input_state, loop_lengths, thetas
    )
    for i, op in enumerate(ph_circuit):
        assert isinstance(op.operation, BS)
        assert isinstance(op.operation, PhotonicGate)
        assert op.qumodes[0]._index == expected_qumodes[i][0]
        assert op.qumodes[1]._index == expected_qumodes[i][1]
        assert op.params[0] == thetas[i]
