# This code is part of qiskit-runtime.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Code taken from qiskit-runtime, with slight modifications was used by PSNC
# for the purpose of creating a QSVM benchmark.

"""The FeatureMap class adapted from qiskit-runtime supplemented with code for preparing MNIST data for QSVM"""

import csv
import os
from importlib.resources import files

import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.parametervector import ParameterVector
from skimage.transform import resize


class FeatureMap:
    """Mapping data with the feature map."""

    def __init__(
        self,
        feature_dimension: int,
        entangler_map: list[list] | None = None,
    ):
        """
        Args:
            feature_dimension (int): number of features, twice the number
                                     of qubits for this encoding
            entangler_map (list[list]): connectivity of qubits with a list of [source, target],
                                        or None for full entanglement. Note that the order in
                                        the list is the order of applying the two-qubit gate.
        Raises:
            ValueError: If the value of ``feature_dimension`` is not an even integer.
        """

        if isinstance(feature_dimension, int):
            if feature_dimension % 2 == 0:
                self._feature_dimension = feature_dimension
            else:
                raise ValueError("Feature dimension must be an even integer.")
        else:
            raise ValueError("Feature dimension must be an even integer.")

        self._num_qubits = int(feature_dimension / 2)

        if entangler_map is None:
            em1 = [[i, i + 1] for i in np.arange(0, self._num_qubits, 2)]
            em2 = [[i - 1, i] for i in np.arange(2, int(self._num_qubits), 2)]
            em3 = [[self._num_qubits - 1, 0]]
            em4 = [
                [i, i + np.sqrt(feature_dimension) / 2]
                for i in np.arange(
                    0,
                    self._num_qubits - np.sqrt(feature_dimension) / 2,
                )
            ]

            em = em1 + em2 + em3 + em4
            em = [[int(element1), int(element2)] for [element1, element2] in em]
            self._entangler_map = em
        else:
            self._entangler_map = entangler_map

        self._num_parameters = self._num_qubits

    def construct_circuit(
        self,
        data: npt.NDArray[np.float64],
        parameters: npt.NDArray[np.float64] | ParameterVector | None = None,
        inverse: bool = False,
        name: str | None = None,
    ):
        """Construct the feature map circuit.

        Args:
            x (numpy.ndarray): data vector of size feature_dimension
            parameters (numpy.ndarray): optional parameters in feature map
            q (QauntumRegister): the QuantumRegister object for the circuit
            inverse (bool): whether or not to invert the circuit
            name (str): name of circuit

        Returns:
            QuantumCircuit: a quantum circuit transforming data x
        Raises:
            ValueError: If the input parameters or vector are invalid
        """

        if isinstance(parameters, np.ndarray):
            if isinstance(parameters, int | float):
                raise ValueError("Parameters must be a list.")
            if len(parameters) == 1:
                parameters = parameters * np.ones(self._num_qubits)
            else:
                if len(parameters) != self._num_parameters:
                    raise ValueError(
                        f"The number of feature map parameters must be {self._num_parameters}"
                    )
        elif parameters is None:
            parameters = ParameterVector("λ", self._num_qubits)

        if len(data) != self._feature_dimension:
            raise ValueError(
                f"The input vector must be of length {self._feature_dimension}."
            )

        q = QuantumRegister(self._num_qubits, name="q")

        circuit = QuantumCircuit(q, name=name)

        for i in range(self._num_qubits):
            circuit.ry(-parameters[i], q[i])  # type: ignore

        for source, target in self._entangler_map:
            circuit.cz(q[source], q[target])

        for i in range(self._num_qubits):
            circuit.rz(-2 * data[2 * i + 1], q[i])
            circuit.rx(-2 * data[2 * i], q[i])

        if inverse:
            return circuit.inverse()
        else:
            return circuit


def prepare_qsvm_circuit(train_data):
    d = train_data.shape[1]
    fm = FeatureMap(feature_dimension=d)

    circuit = fm.construct_circuit(data=train_data[0])

    circuit.measure_all()

    return circuit


def sort_2_arrays(list1, list2):
    p = list1.argsort()
    return list1[p], list2[p]


def load_csv_data(path: str):
    data = []
    with open(path, encoding="UTF-8") as csvfile:
        reader_variable = csv.reader(csvfile, delimiter=",")
        for row in reader_variable:
            data.append(row)
    return np.array(data, dtype=np.int16)


def load_prepared_mnist(
    file: str, train_size: int = 20, img_dim=3, seed: int | None = None
):
    data = load_csv_data(file)
    x_train = data[:, 1:]
    y_train = data[:, 0]
    np.random.seed(seed)
    train_shuffler = np.random.permutation(len(y_train))
    x_train = x_train[train_shuffler][:train_size]
    y_train = y_train[train_shuffler][:train_size]

    x_train = x_train.reshape(-1, 28, 28) / 255
    x_train = np.array(
        [resize(img, (img_dim, img_dim), anti_aliasing=True) for img in x_train]
    )
    x_train = x_train.reshape(-1, img_dim * img_dim)

    y_train, x_train = sort_2_arrays(y_train, x_train)

    y_train = (y_train * 2) - 1

    return x_train, y_train


def trained_qsvm_8q() -> tuple[QuantumCircuit, tuple[float, ...]]:
    datafile = files("open_qbench.data").joinpath("mnist_train100.csv")
    train_data_x, _ = load_prepared_mnist(datafile, 20, 4, seed=123)
    circuit = prepare_qsvm_circuit(train_data_x)
    circuit.name = "QSVM_MNIST_8q"
    parameters = [
        0.61069116,
        1.99988148,
        1.05376726,
        0.33253631,
        1.53000978,
        1.38195184,
        2.43730072,
        0.21604097,
    ]
    return circuit, parameters


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    qc, params = trained_qsvm_8q()

    from qiskit.primitives import Sampler

    res = Sampler().run(qc, params).result().quasi_dists
    print(res)
