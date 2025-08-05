"""Interfaces and classes used for working with PhotonicCircuit"""

import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence

from qiskit.circuit import Bit, Register


class Qumode(Bit):
    """A quantum mode -- a fundamental unit of information for CV quantum computers.

    Inheriting from :class:'Bit' might be confusing since qumodes are continuous, but
    it doesn't enforce constraints on the underlying data and provides some
    useful methods for working with :class:'Register'.
    """


class PhotonicRegister(Register):
    """A register holding qumodes.
    Analogous to :class:'QuantumRegister'.
    """

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    prefix = "qm"  # Prefix to use for auto naming.
    bit_type = Qumode


class PhotonicOperation(ABC):
    """This interface mirrors qiskit.circuit.operation.Operation, but instead of num_qubits,
    we have num_qumodes and instead of num_clbits, we have num_clints, which are used
    to count the photons.

    This interface is used directly by :class:'PhotonicOperation'.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def name(self):
        """Unique string identifier for operation type."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_qumodes(self):
        """Number of qumodes."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_clints(self):
        """Number of classical integers to store photon counts."""
        raise NotImplementedError


class PhotonicInstruction(PhotonicOperation):
    """A generic photonic instruction based on :class:'qiskit.circuit.instruction.Instruction'.
    It can describe both unitary operations (gates) and non-unitary operations (measurements).

    They can be tied to hardware implementation, hence the 'duration' and 'unit' attributes.

    Same as the original :class:'Instruction', these instructions do not have any context
    about where they are in a circuit. This is handled by the circuit itself.
    """

    def __init__(
        self,
        name: str,
        num_qumodes: int,
        num_clints: int,
        params: Sequence[float],
        duration: int | float | None = None,
        unit: str = "dt",
        label: str | None = None,
    ):
        self._name = name
        self._num_qumodes = num_qumodes
        self._num_clints = num_clints
        self.duration = duration
        self.unit = unit
        self.label = label
        self.params = params

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def num_qumodes(self):
        return self._num_qumodes

    @num_qumodes.setter
    def num_qumodes(self, num_qumodes):
        self._num_qumodes = num_qumodes

    @property
    def num_clints(self):
        return self._num_clints

    @num_clints.setter
    def num_clints(self, num_clints):
        self._num_clints = num_clints

    # TODO: implement methods from Instruction, that also apply to this class


class PhotonicCircuitInstruction:
    """Used to tie a :class:'PhotonicGate' to qumodes in a PhotonicRegister.
    #TODO: generalize to PhotonicInstruction to also include measurements and clints.

    Qiskit currently uses Rust to implement a :class:'CircuitInstruction'.
    When instantiated in Python, these objects look like this:
    .. code-block:: python
        `CircuitInstruction(operation=Instruction(name='h', num_qubits=1,
        num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 0),), clbits=())`

    This class is a simplified analogue of the corresponding Rust struct found in `qiskit._accelerate'.

    """

    def __init__(
        self,
        operation: PhotonicInstruction,
        qumodes: Sequence[Qumode],
    ):
        self.operation = operation
        self.qumodes = qumodes
        self.params = operation.params


class PhotonicGate(PhotonicInstruction):
    """A unitary gate acting on qumodes.
    Based on :class:'qiskit.circuit.gate.Gate'.
    """

    def __init__(
        self,
        name: str,
        num_qumodes: int,
        params: list,
        label: str | None = None,
        duration=None,
        unit="dt",
    ):
        super().__init__(name, num_qumodes, 0, params, duration, unit, label)

    def __repr__(self) -> str:
        """Generates a representation of the PhotonicGate object instance
        Returns:
            str: A representation of the PhotonicGate instance with the name,
                 number of qumodes, classical bits and params( if any )
        """
        return (
            f"PhotonicGate(name='{self.name}', num_qumodes={self.num_qumodes}, "
            f"params={self.params})"
        )

    @property
    def num_qumodes(self):
        return self._num_qumodes

    @num_qumodes.setter
    def num_qumodes(self, num_qumodes):
        """Set num_qubits."""
        self._num_qumodes = num_qumodes

    def validate_operands(self, qumodes):
        for qumode in qumodes:
            if not isinstance(qumode, Qumode):
                raise TypeError("A photonic gate can only be applied to Qumodes")


class BS(PhotonicGate):
    """A beamsplitter gate"""

    def __init__(
        self,
        theta: float,
        label: str | None = None,
        *,
        duration=None,
        unit="dt",
    ):
        """Create new BS gate."""
        super().__init__(
            "bs",
            2,
            [theta],
            label=label,
            duration=duration,
            unit=unit,
        )

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the Beamsplitter gate."""
        if copy is False:
            raise ValueError(
                "unable to avoid copy while creating an array as requested"
            )

        # return numpy.array([], dtype=dtype)

    def __eq__(self, other):
        if isinstance(other, BS):
            return self._compare_parameters(
                other
            )  # TODO Fix: there is no such method in BS, define it as abstract or implement
        return False

    def _compare_parameters(self, other):
        return isinstance(other, BS) and self.params[0] == other.params[0]

    def _define(self):
        # define decomposition, if needed
        pass
