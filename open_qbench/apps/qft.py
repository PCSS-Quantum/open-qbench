import math

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT


def prepare_QFT(encoded_number: int):
    n_qubits = len(bin(encoded_number)) - 2
    q = QuantumRegister(n_qubits, "q")

    circuit = QuantumCircuit(q)
    circuit.h(q)
    for i, qubit in enumerate(q):
        angle = encoded_number * math.pi / 2**i
        circuit.rz(angle, qubit)

    circuit &= QFT(
        num_qubits=n_qubits,
        approximation_degree=0,
        do_swaps=False,
        inverse=True,
        insert_barriers=True,
        name="qft",
    )
    circuit.name = f"QFT_{n_qubits}q"

    return circuit
