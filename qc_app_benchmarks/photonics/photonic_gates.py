import itertools
from abc import ABC, abstractmethod
from typing import Sequence

from qiskit.circuit import Bit, ParameterExpression, Register


class Qumode(Bit):
    """Implement a quantum mode"""

    pass


class PhotonicRegister(Register):
    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    prefix = "qm"  # Prefix to use for auto naming.
    bit_type = Qumode


class PhotonicOperation(ABC):
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


class PhotonicCircuitInstruction:
    """Used to tie a PhotonicGate to qumodes in a PhotonicRegister"""

    def __init__(
        self,
        operation: PhotonicInstruction,
        qumodes: Sequence[Qumode],
    ):
        self.operation = operation
        self.qumodes = qumodes
        self.params = operation.params


class PhotonicGate(PhotonicInstruction):
    """A photonic gate acting on qumodes"""

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
                raise TypeError(f"A photonic gate can only be applied to Qumodes")


class BS(PhotonicGate):
    """A beamsplitter gate"""

    def __init__(
        self,
        theta: ParameterExpression,
        float,
        label: str | None = None,
        *,
        duration=None,
        unit="dt",
    ):
        """Create new BS gate."""
        super().__init__("bs", 2, [theta], label=label, duration=duration, unit=unit)

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the Beamsplitter gate."""
        if copy is False:
            raise ValueError(
                "unable to avoid copy while creating an array as requested"
            )

        # return numpy.array([], dtype=dtype)

    def __eq__(self, other):
        if isinstance(other, BS):
            return self._compare_parameters(other)
        return False

    def _define(self):
        # define decomposition, if needed
        pass
