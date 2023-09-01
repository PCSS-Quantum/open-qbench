from qiskit import QuantumCircuit


def grover_3q() -> QuantumCircuit:
    """Resturns a 3-qubit Grover circuit with one marked state (111)"""
    circuit = QuantumCircuit(3, 3)
    circuit.h(range(3))
    circuit.ccz(0, 1, 2)
    circuit.h(range(3))
    circuit.x(range(3))
    circuit.ccz(0, 1, 2)
    circuit.x(range(3))
    circuit.h(range(3))
    circuit.measure(range(3), range(3))

    circuit.name = "grover_3q"
    return circuit


if __name__ == "__main__":
    qc = grover_3q()
    # qc.draw("latex", filename=qc.name + ".png")
    from qiskit.primitives import Sampler

    res = Sampler().run(qc).result().quasi_dists
    print(res)
