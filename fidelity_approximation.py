'''
Starter code to test approximation using fidelity metric
DO NOT USE THIS! USE transformation_matrix_approximation.py INSTEAD! 
'''

from qiskit import QuantumCircuit, ClassicalRegister
import qiskit
import random
import numpy as np
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import random_statevector

# Set random seed for reproducibility
# random.seed(10)

# Define initial state
initial_state = random_statevector(2)

# Make list of equivalent gates
single_qubit_gates = ["x", "y", "z", "h", "s", "t", "id",
                          "sxdg", "sdg", "sx", "tdg"]
# single_qubit_gates = ["x", "y", "z"]


# Create original quantum circuit with one qubit
qc_original = QuantumCircuit(1)
# qc_original.initialize(initial_state, 0)
angle = random.uniform(np.pi*0, np.pi*2)
# angle = 1.8878
qc_original.rx(angle, 0)
# qc_original = random_circuit(1, 20)
classical_register_orig = ClassicalRegister(1)

backend = qiskit.Aer.get_backend('statevector_simulator')
result_original = qiskit.execute(qc_original, backend).result()
statevector_original = result_original.get_statevector(qc_original)
print("Original Statevector:", statevector_original)

dm_original = qiskit.quantum_info.DensityMatrix(qc_original)

# Create new quantum circuit with one qubit
qc_new = QuantumCircuit(1, 1)
# qc_new.initialize(initial_state, 0)


# Apply iterative fidelity improvement greedily 
fidelity = 0
prev = 'null'
for i in range(5):
    fid_dict = {}
    for gate in single_qubit_gates:
        fid_dict[gate] = 0.0
    for gate in single_qubit_gates:
        if gate == prev:
            # print("here", gate)
            continue
        getattr(qc_new, gate)(0)
        result_new = qiskit.execute(qc_new, backend).result()
        statevector_new = result_new.get_statevector(qc_new)
        print(gate, statevector_new)
        fid_dict[gate] = qiskit.quantum_info.state_fidelity(statevector_original, statevector_new)
        del qc_new.data[-1]
    # print(fid_dict)
    sorted_fid_dict = {k: v for k, v in sorted(fid_dict.items(), key=lambda item: item[1], reverse=True)}
    top_gates = list(sorted_fid_dict.keys())[:4]
    # best_gate = max(fid_dict, key=fid_dict.get)
    best_gate = random.choice(top_gates)
    prev = best_gate
    # print(prev)
    if fidelity < fid_dict[best_gate]:
        fidelity = fid_dict[best_gate]
        getattr(qc_new, best_gate)(0)
    else:
        continue
    # fidelity = fid_dict[best_gate]
    # getattr(qc_new, best_gate)(0)
    print(statevector_original)
    print(fidelity)
    print(qc_original)
    print(qc_new)

# orig_inverse = qc_original.inverse()
# qc_combined = QuantumCircuit(1,1)
# qc_combined.compose(qc_new, inplace=True)
# qc_combined.compose(orig_inverse, inplace=True)
# qc_combined.measure(0,0)
# # qc_new.compose(orig_inverse, inplace=True)
# qc_new.measure(0, 0)
# qc_original.add_register(classical_register_orig)
# qc_original.measure(0,0)
# backend_sim = qiskit.Aer.get_backend('qasm_simulator')

# job_orig = backend_sim.run(qc_original, shots=8192)
# result_sim = job_orig.result()
# counts = result_sim.get_counts(qc_original)
# print(counts)

# job_new = backend_sim.run(qc_new, shots=8192)
# result_sim = job_new.result()
# counts = result_sim.get_counts(qc_new)
# print(counts)

# job_combined = backend_sim.run(qc_combined, shots=8192)
# result_sim = job_combined.result()
# counts = result_sim.get_counts(qc_combined)
# print(counts)

