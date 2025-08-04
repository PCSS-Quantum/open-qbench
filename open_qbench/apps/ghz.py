import math

from qiskit import QuantumCircuit, QuantumRegister


def ghz_direct(num_qubits: int) -> QuantumCircuit:
    reg = QuantumRegister(size=num_qubits)
    circuit = QuantumCircuit(reg)

    for i in range(num_qubits - 1):
        circuit.rxx(math.pi / 2, 0, i + 1)
        circuit.rx(math.pi / 2, i + 1)
    circuit.ry(math.pi / 2, 0)
    circuit.measure_all()
    return circuit


def ghz_decoherence_free(num_qubits: int) -> QuantumCircuit:
    reg = QuantumRegister(size=num_qubits)
    circuit = QuantumCircuit(reg)
    circuit.sx(0)

    for qubit in range(num_qubits):
        if qubit % 2 == 1:
            circuit.x(qubit)

    circuit.barrier(list(range(num_qubits)))
    for target_qubit in range(num_qubits):
        if target_qubit > 0:
            circuit.cx(0, target_qubit)
    circuit.barrier(list(range(num_qubits)))

    for qubit in range(num_qubits):
        if qubit % 2 == 1:
            circuit.x(qubit)
    circuit.barrier(list(range(num_qubits)))
    return circuit
