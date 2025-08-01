from qiskit import QuantumCircuit


def toffoli_circuit(
    num_qubits: int = 5, input_state: int | str = "11111"
) -> QuantumCircuit:
    """Returns an n-qubit Toffoli circuit with specified input state"""
    circuit = QuantumCircuit(num_qubits)
    if isinstance(input_state, int):
        input_state = bin(input_state)[2:]
    for i, q in enumerate(reversed(input_state)):
        if int(q) == 1:
            circuit.x(i)
    circuit.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    circuit.measure_all()

    circuit.name = f"Toffoli_{num_qubits}q"
    return circuit
