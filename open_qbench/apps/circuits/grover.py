import math

from qiskit import QuantumCircuit


def grover_nq(num_qubits: int, marked_state: int | str) -> QuantumCircuit:
    """Resturns an n-qubit Grover circuit with one marked state"""
    if isinstance(marked_state, int):
        marked_state = bin(marked_state)[2:]
    if len(marked_state) > num_qubits:
        raise ValueError(
            "Number of bits in the marked state cannot be larger than number of qubits"
        )
    marking_circ = QuantumCircuit(num_qubits, num_qubits)
    for i, q in enumerate(reversed(marked_state)):
        if int(q) == 0:
            marking_circ.x(i)

    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))
    circuit &= marking_circ
    circuit.mcp(math.pi, list(range(num_qubits - 1)), num_qubits - 1)
    circuit &= marking_circ
    circuit.h(range(num_qubits))
    circuit.x(range(num_qubits))
    circuit.mcp(math.pi, list(range(num_qubits - 1)), num_qubits - 1)
    circuit.x(range(num_qubits))
    circuit.h(range(num_qubits))

    circuit.name = f"Grover_{num_qubits}q"
    return circuit
