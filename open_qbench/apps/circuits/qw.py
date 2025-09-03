# Classes:
#   -QW_Helpers
#        Functions:
#           -display_res_with_coin(data,begin, end, title)
#           -display_res_without_coin(data, begin, end, title)
#           -to_ket(number,no_qubits)
#           -to_bra(number,no_qubits)
#
#   -QW_Cycle:
#       Functions:
#           -create_c_one_gate(self, quantum_circuit,first_qubit_id,
#                              target_qubit_id):
#           -create_inc_One_gate(self,quantum_circuit, no_qubits,
#                                first_qubit_id)
#           -create_control_zero_gate(self, quantum_circuit,
#                                first_qubit_id,
#                                target_qubit_id)
#           -create_dec_One_gate(self,quantum_circuit,no_qubits,
#                                first_qubit_id)
#           -apply_step(self,quantum_circuit, no_qubits)
#           -run_dtqw(self, no_qubits,steps,position,no_attempts)
#
# Qiskit Version:
#
#
#


import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation


class QW_Helpers:
    k_zero = np.array([1, 0])[:, np.newaxis]
    k_one = np.array([0, 1])[:, np.newaxis]
    b_zero = k_zero.transpose()
    b_one = k_one.transpose()

    def display_res_with_coin(self, data, begin, end, title):
        new_data = {}
        for key, value in data.items():
            shortened_key = int(key[1:], 2)
            if shortened_key in new_data:
                new_data[shortened_key] += value
            else:
                new_data[shortened_key] = value
        x = []
        y = []
        for pos in range(begin, end):
            ys = 0.0
            if pos in new_data:
                ys = new_data[pos]
            y.append(ys)
            x.append(pos)
        plt.title(title)
        plt.xlabel("Position")
        plt.ylabel("No times walker was at position")
        plt.plot(x, y)
        plt.show()

    def display_res_without_coin(self, data, begin, end, title):
        new_data = {}
        for key, value in data.items():
            shortened_key = int(key, 2)
            if shortened_key in new_data:
                new_data[shortened_key] += value
            else:
                new_data[shortened_key] = value
        x = []
        y = []
        for pos in range(begin, end):
            ys = 0.0
            if pos in new_data:
                ys = new_data[pos]
            y.append(ys)
            x.append(pos)
        plt.title(title)
        plt.xlabel("Position")
        plt.ylabel("No times walker was at position")
        plt.plot(x, y)
        plt.show()

    def to_ket(self, number: np.uint16, no_qubits):
        binary = None
        first = None
        if number >= 0:
            binary = (bin(number)[2:]).zfill(no_qubits)
            first = self.k_one if binary[0] == "1" else self.k_zero
        res = first
        for i in range(1, no_qubits):
            next_p = None
            next_p = self.k_one if binary[i] == "1" else self.k_zero
            res = np.kron(res, next_p)
        return res

    def to_bra(self, number: np.uint16, no_qubits):
        first = None
        binary = (bin(number)[2:]).zfill(no_qubits)
        first = self.b_one if binary[0] == "1" else self.b_zero
        res = first
        for i in range(1, no_qubits):
            next_p = self.b_one if binary[i] == "1" else self.b_zero
            res = np.kron(res, next_p)
        return res


class QW_Cycle:
    def create_c_one_gate(self, quantum_circuit, first_qubit_id, target_qubit_id):
        if target_qubit_id == first_qubit_id:
            quantum_circuit.x(target_qubit_id)
        else:
            control_qubits = [i for i in range(first_qubit_id, target_qubit_id)]
            quantum_circuit.mcx(control_qubits, target_qubit_id)
        return quantum_circuit

    def create_inc_One_gate(self, quantum_circuit, no_qubits, first_qubit_id):
        for i in range(first_qubit_id + 1, no_qubits + 1):
            target_qubit_id = first_qubit_id - i + no_qubits
            quantum_circuit = self.create_c_one_gate(
                quantum_circuit, first_qubit_id, target_qubit_id
            )
        return quantum_circuit

    def create_control_zero_gate(
        self, quantum_circuit, first_qubit_id, target_qubit_id
    ):
        if target_qubit_id == first_qubit_id:
            quantum_circuit.x(target_qubit_id)
        else:
            control_qubits = [i for i in range(first_qubit_id, target_qubit_id)]
            quantum_circuit.x(control_qubits)
            quantum_circuit.mcx(control_qubits, target_qubit_id)
            quantum_circuit.x(control_qubits)
        return quantum_circuit

    def create_dec_One_gate(self, quantum_circuit, no_qubits, first_qubit_id):
        for i in range(first_qubit_id + 1, no_qubits + 1):
            target_qubit_id = first_qubit_id - i + no_qubits
            quantum_circuit = self.create_control_zero_gate(
                quantum_circuit, first_qubit_id, target_qubit_id
            )
        return quantum_circuit

    def set_position(self, initial_state, quantum_circuit):
        for qubit_id, qubit_value in enumerate(initial_state):
            if qubit_value == 1:
                binary = bin(qubit_id)[2:]
                for i, q in enumerate(binary):
                    if q == "1":
                        quantum_circuit.x(i)
        return quantum_circuit

    def create_dtqw_circuit(self, no_qubits, steps, position):
        q_ids = [i for i in range(no_qubits)]
        quantum_circuit = QuantumCircuit(no_qubits)
        initial_state = [q for vector in position for q in vector]
        prep = StatePreparation(initial_state)
        quantum_circuit.append(prep, q_ids)

        qubits = [i for i in range(0, no_qubits - 1)]
        qubits = [no_qubits - 1, *qubits]
        empty_circ = QuantumCircuit(no_qubits - 1)
        inc_gate = (
            self.create_inc_One_gate(empty_circ.copy(), no_qubits - 1, 0)
            .to_gate()
            .control(1)
        )
        dec_gate = (
            self.create_dec_One_gate(empty_circ.copy(), no_qubits - 1, 0)
            .to_gate()
            .control(1)
        )

        for _ in range(steps):
            quantum_circuit.h(no_qubits - 1)
            quantum_circuit.append(inc_gate, qubits)
            quantum_circuit.x(no_qubits - 1)
            quantum_circuit.append(dec_gate, qubits)
            quantum_circuit.x(no_qubits - 1)
        return quantum_circuit
