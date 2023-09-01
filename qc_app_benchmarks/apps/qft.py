from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import QFT
import math


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
    circuit.measure_all()
    circuit.name = f"QFT_{n_qubits}q"

    return circuit


if __name__ == "__main__":
    qc = prepare_QFT(24)

    from qiskit.primitives import Sampler

    res = Sampler().run(qc).result().quasi_dists[0]
    print(res)
