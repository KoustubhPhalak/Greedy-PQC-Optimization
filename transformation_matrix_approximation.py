'''
Contains main functions to transform a quantum circuit based on
the distance metric between transformation matrices
'''

from qiskit import QuantumCircuit, ClassicalRegister
import qiskit
import random
import numpy as np
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import random_statevector

# Make list of equivalent gates
single_qubit_gates = ["x", "y", "z", "h", "s", "t", "id",
                          "sxdg", "sdg", "sx", "tdg"]

def hs_dist_loss(matrix_1, matrix_2):
    '''
    Calculates the Hilbert-Schmidt distance between two 2x2 unitary matrices.
    matrix_1: reconstructed transformation matrix
    matrix_2: original transformation matrix
    '''
    return 1 - np.abs(np.trace(np.matmul(np.conj(np.transpose(matrix_1)), matrix_2)))/2

def transform_qc(qc_original):
    qc_new = QuantumCircuit(1, 1)
    tm_original = qiskit.quantum_info.Operator(qc_original)
    final_dist = 1000
    prev = 'null'
    for i in range(20):
        dist_dict = {}
        for gate in single_qubit_gates:
            dist_dict[gate] = 1000
        for gate in single_qubit_gates:
            if gate == prev:
                continue
            getattr(qc_new, gate)(0)
            tm_new = qiskit.quantum_info.Operator(qc_new)
            dist_dict[gate] = hs_dist_loss(tm_new.data, tm_original.data)
            del qc_new.data[-1]
        sorted_dist_dict = {k: v for k, v in sorted(dist_dict.items(), key=lambda item: item[1])}
        top_gates = list(sorted_dist_dict.keys())[:4]
        best_gate = random.choice(top_gates)
        best_gate = min(dist_dict, key=dist_dict.get)
        if final_dist > dist_dict[best_gate]:
            prev = best_gate
            final_dist = dist_dict[best_gate]
            getattr(qc_new, best_gate)(0)
        else:
            continue
        # print(qc_original)
        # print(qc_new)
        # print(final_dist)
    # print("Final dist,", final_dist)
    return final_dist, qc_new


