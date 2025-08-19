from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp


def jssp_7q_24d() -> tuple[QuantumCircuit, tuple[float, ...]]:
    """Returns a 7-qubit QAOA circuit with normalized depth 24, as in the document.
    Returned parameters are chosen arbitrarily, so that the final distribution is
    not uniform.
    """
    hamiltonian = SparsePauliOp.from_list(
        [
            ("IIIIIII", (3.125 + 0j)),
            ("IIIIIIZ", (-0.25 + 0j)),
            ("IIIIIZI", (0.25 + 0j)),
            ("IIZIIII", (-0.625 + 0j)),
            ("IIIZIII", (-0.375 + 0j)),
            ("IIZZIII", (0.125 + 0j)),
            ("IIIIZII", (-0.125 + 0j)),
            ("IIZIZII", (0.125 + 0j)),
            ("IIIZZII", (0.125 + 0j)),
            ("IIZZZII", (0.375 + 0j)),
            ("ZZIIIII", (0.5 + 0j)),
            ("IIZIIIZ", (0.5 + 0j)),
            ("IIIZIIZ", (0.25 + 0j)),
            ("ZIIIIII", (-0.25 + 0j)),
            ("ZIIIIZI", (0.25 + 0j)),
        ]
    )
    qc = QAOAAnsatz(hamiltonian, reps=1)
    params = (0.388917, 5.44861221)
    qc.name = "QAOA_JSSP_7q"

    return qc, params
