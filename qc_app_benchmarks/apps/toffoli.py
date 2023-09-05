from qiskit import QuantumCircuit


def toffoli_circuit(
    num_qubits: int = 5, input_state: int | str = "11111"
) -> QuantumCircuit:
    """Returns an n-qubit Toffoli circuit with specified input state"""
    circuit = QuantumCircuit(num_qubits, num_qubits)
    if isinstance(input_state, int):
        input_state = bin(input_state)[2:]
    for i, q in enumerate(reversed(input_state)):
        if int(q) == 1:
            circuit.x(i)
    circuit.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    circuit.measure(range(num_qubits), range(num_qubits))

    circuit.name = f"toffoli_{num_qubits}q"
    return circuit


if __name__ == "__main__":
    qc = toffoli_circuit(5, "11111")
    from qiskit.primitives import Sampler

    res = Sampler().run(qc).result().quasi_dists[0].binary_probabilities()
    print(res)
