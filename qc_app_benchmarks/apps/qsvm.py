import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.parametervector import ParameterVector

from skimage.transform import resize
import pandas as pd


class FeatureMap3:
    """Mapping data with the feature map."""

    def __init__(self, feature_dimension, entangler_map=None):
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
                for i in np.arange(0, self._num_qubits - np.sqrt(feature_dimension) / 2)
            ]

            em = em1 + em2 + em3 + em4
            em = [[int(element1), int(element2)] for [element1, element2] in em]
            self._entangler_map = em
            # self._entangler_map = [
            #     [i, j]
            #     for i in range(self._feature_dimension)
            #     for j in range(i + 1, self._feature_dimension)
            # ]
        else:
            self._entangler_map = entangler_map

        self._num_parameters = self._num_qubits

    def construct_circuit(
        self, x=None, parameters=None, q=None, inverse=False, name=None
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

        if parameters is not None:
            if isinstance(parameters, (int, float)):
                raise ValueError("Parameters must be a list.")
            if len(parameters) == 1:
                parameters = parameters * np.ones(self._num_qubits)
            else:
                if len(parameters) != self._num_parameters:
                    raise ValueError(
                        "The number of feature map parameters must be {}.".format(
                            self._num_parameters
                        )
                    )
        else:
            parameters = ParameterVector("λ", self._num_qubits)

        if len(x) != self._feature_dimension:
            raise ValueError(
                "The input vector must be of length {}.".format(self._feature_dimension)
            )

        if q is None:
            q = QuantumRegister(self._num_qubits, name="q")

        circuit = QuantumCircuit(q, name=name)

        for i in range(self._num_qubits):
            circuit.ry(-parameters[i], q[i])

        for source, target in self._entangler_map:
            circuit.cz(q[source], q[target])

        for i in range(self._num_qubits):
            circuit.rz(-2 * x[2 * i + 1], q[i])
            circuit.rx(-2 * x[2 * i], q[i])

        if inverse:
            return circuit.inverse()
        else:
            return circuit


class FeatureMap:
    """Mapping data with the feature map."""

    def __init__(self, feature_dimension, entangler_map=None):
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
            em = em1 + em2 + em3
            em = [[int(element1), int(element2)] for [element1, element2] in em]
            self._entangler_map = em
            # self._entangler_map = [
            #     [i, j]
            #     for i in range(self._feature_dimension)
            #     for j in range(i + 1, self._feature_dimension)
            # ]
        else:
            self._entangler_map = entangler_map

        self._num_parameters = self._num_qubits

    def construct_circuit(
        self, x=None, parameters=None, q=None, inverse=False, name=None
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

        if parameters is not None:
            if isinstance(parameters, (int, float)):
                raise ValueError("Parameters must be a list.")
            if len(parameters) == 1:
                parameters = parameters * np.ones(self._num_qubits)
            else:
                if len(parameters) != self._num_parameters:
                    raise ValueError(
                        "The number of feature map parameters must be {}.".format(
                            self._num_parameters
                        )
                    )
        else:
            parameters = ParameterVector("λ", self._num_qubits)

        if len(x) != self._feature_dimension:
            raise ValueError(
                "The input vector must be of length {}.".format(self._feature_dimension)
            )

        if q is None:
            q = QuantumRegister(self._num_qubits, name="q")

        circuit = QuantumCircuit(q, name=name)

        for i in range(self._num_qubits):
            circuit.ry(-parameters[i], q[i])

        for source, target in self._entangler_map:
            circuit.cz(q[source], q[target])

        for i in range(self._num_qubits):
            circuit.rz(-2 * x[2 * i + 1], q[i])
            circuit.rx(-2 * x[2 * i], q[i])

        if inverse:
            return circuit.inverse()
        else:
            return circuit


def prepare_qsvm_circuit(train_data):
    d = train_data.shape[1]
    fm = FeatureMap3(feature_dimension=d)

    qc = fm.construct_circuit(x=train_data[0])
    # ) & fm.construct_circuit(x=train_data_x[1], parameters=initial_point, inverse=True)

    qc.measure_all()

    return qc


def sort_2_arrays(list1, list2):
    p = list1.argsort()
    return list1[p], list2[p]


def load_prepared_mnist(train_size: int = 20, seed: int = None):
    img_dim = 4

    import os

    dirname = os.path.dirname(__file__)

    x_train = np.array(
        pd.read_csv(os.path.join(dirname, "../data/mnist_train100_dim4.csv"))
    )
    y_train = np.array(
        pd.read_csv(os.path.join(dirname, "../data/mnist_ytrain100_dim4.csv"))
    )[:, 0]

    np.random.seed(seed)
    train_shuffler = np.random.permutation(len(y_train))
    x_train = x_train[:][:train_size]
    y_train = y_train[:][:train_size]

    x_train = x_train.reshape(-1, 28, 28) / 255
    x_train = np.array(
        [resize(img, (img_dim, img_dim), anti_aliasing=True) for img in x_train]
    )
    x_train = x_train.reshape(-1, img_dim * img_dim)

    y_train, x_train = sort_2_arrays(y_train, x_train)

    y_train = (y_train * 2) - 1

    return x_train, y_train


def trained_qsvm_8q():
    train_data_x, _ = load_prepared_mnist(2, seed=123)
    qc = prepare_qsvm_circuit(train_data_x)
    qc.name = "QSVM_MNIST_8q"
    params = [
        0.61069116,
        1.99988148,
        1.05376726,
        0.33253631,
        1.53000978,
        1.38195184,
        2.43730072,
        0.21604097,
    ]
    return qc, params


def trained_qsvm_16q():
    d = 36
    # np.random.seed(345)
    params = [np.random.uniform(0, np.pi) for _ in range(d // 2)]
    qc = prepare_qsvm_circuit(d)
    return qc, params


# qc, params = trained_qsvm_8q()
# print(qc)
# qc.draw(output="latex", filename="QSVM_quantum_cirtcuit2.png")

# from qiskit.primitives import Sampler
# from qiskit.visualization import plot_histogram

# res = Sampler().run(qc, params).result().quasi_dists
# plot_histogram(res, filename="qsvm.png")
