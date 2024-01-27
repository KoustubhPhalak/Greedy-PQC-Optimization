'''
Transforms the trained model using the distance metric
Uses trained model from qnn_train_basic_entangler.py
and transformation function from transformation_matrix_approximation.py
'''

from transformation_matrix_approximation import *
import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, ClassicalRegister, transpile
import qiskit
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Define the quantum device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# For iris, 8 qubits, For digits, 10 qubits
qubits = 8
dev = qml.device("lightning.qubit", wires=qubits)

# Define the PQC circuit
@qml.qnode(dev)
def pqc_circuit_iris_new(inputs, params):
    # qml.templates.AngleEmbedding(inputs, wires=range(4))
    qml.templates.BasicEntanglerLayers(params, wires=range(qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

@qml.qnode(dev)
def pqc_iris_strong_new(inputs, params):
    # qml.templates.AngleEmbedding(inputs, wires=range(4))
    qml.templates.StronglyEntanglingLayers(params, wires=range(qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

@qml.qnode(dev)
def pqc_circuit_digits_new(inputs, params):
    # qml.templates.AmplitudeEmbedding(inputs, wires=range(6), normalize=True)
    qml.templates.BasicEntanglerLayers(params, wires=range(qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

@qml.qnode(dev)
def pqc_digits_strong_new(inputs, params):
    # qml.templates.AmplitudeEmbedding(inputs, wires=range(6), normalize=True)
    qml.templates.StronglyEntanglingLayers(params, wires=range(qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"params": (5, qubits)}
        weight_shapes_strong = {"params": (5, qubits, 3)}
        self.pqc = qml.qnn.TorchLayer(pqc_iris_strong_new, weight_shapes_strong, init_method=torch.nn.init.normal_)

    def forward(self, x):
        return self.pqc(x)
    
model = Model().to(device)

def get_trained_pqc_qasm(model_path="model_params_iris_strong.pth"): # Don't forget to change model path here!
    '''
    Imports model params and fits them in PQC, converts it to QASM, and returns the QASM code.
    '''
    # Load saved parameters from file
    params = torch.load(model_path)
    # model.load_state_dict(torch.load("model_params.pth"))
    pqc_iris_strong_new.construct([0.5, params['pqc.params'].tolist()], {})
    t = pqc_iris_strong_new.qtape.to_openqasm()

    # Check original transpiled circuit depth and gate count
    qc = QuantumCircuit.from_qasm_str(t)
    qc_transpiled = transpile(qc, basis_gates=['sx', 'cx', 'rz', 'id', 'x'], optimization_level=2)
    print(f"Original circuit data: Depth:{qc_transpiled.depth()}, Gate count: {qc_transpiled.count_ops()}")


    # Split the QASM code into lines
    lines = t.split('\n')

    # Filter out lines containing 'measure' operation
    lines_without_measurements = [line for line in lines if not line.strip().startswith('measure')]

    # Join the remaining lines back into a single string
    modified_t = '\n'.join(lines_without_measurements)

    # print(modified_t)
    return modified_t

def parse_qasm_file(qasm_file):
    '''
    Take QASM file as input and
    1. Parse the QASM file into a list, such that the list contains:
        a. The gate name
        b. Qubit information
        c. Parametric angle (if rotation gate)
    2. Return the parsed list
    '''

    parsed_list = []
    lines = qasm_file.split('\n')
    for line in lines:
        parts = line.split()
        if not parts:
            continue

        gate_name = parts[0].split('(')[0]

        if gate_name in ["rx", "ry", "rz"]:
            angle = float(parts[0].split('(')[1].rstrip(')'))
            qubit_number = int(parts[1].strip('q[];'))
            parsed_list.append({'gate_name': gate_name, 'angle': angle, 'qubit_number': qubit_number})
        elif gate_name in ["cx", "ccx"]:
            qubits = parts[1].split(',')
            qubit_numbers = [int(qubit.strip('q[];')) for qubit in qubits]
            parsed_list.append({'gate_name': gate_name, 'angle': None, 'qubit_number': qubit_numbers})
        elif gate_name in ["x", "y", "z", "h", "id", "s", "t", "sx", "sxdg", "tdg", "sdg"]:
            qubit_number = int(parts[1].strip('q[];'))
            parsed_list.append({'gate_name': gate_name, 'angle': None, 'qubit_number': qubit_number})

    return parsed_list

def pqc_greedy_optimization(parsed_list, etol=0.005):
    '''
    Take parsed gate list as input and
    1. Create a new empty qiskit circuit, 
    2. Perform greedy optimization of RX gates such that if distance is less than etol,
    then keep the optimized version, else retain the original gate(s)
    3. Add the chosen (optimized/retained) version of gate to the empty circuit
    4. Keep on doing until all RX gates are optimized/retained
    5. finally return the optimized circuit
    '''

    qc_reconstructed = QuantumCircuit(qubits)
    for gate in parsed_list:
        if gate['gate_name'] == 'cx':
            qc_reconstructed.cx(gate['qubit_number'][0], gate['qubit_number'][1])
        elif gate['gate_name'] == 'rx' or gate['gate_name'] == 'ry' or gate['gate_name'] == 'rz':
            orig_gate = QuantumCircuit(1)
            getattr(orig_gate, gate['gate_name'])(gate['angle'], 0)
            # orig_gate.rx(gate['angle'], 0)
            hs_dist, recon_gate = transform_qc(orig_gate)
            if hs_dist < etol:
                qc_reconstructed = qc_reconstructed.compose(recon_gate, qubits=[gate['qubit_number']])
            else:
                qc_reconstructed = qc_reconstructed.compose(orig_gate, qubits=[gate['qubit_number']])
    return qc_reconstructed

def add_pennylane_gates(qasm_string, params=None):
    '''
    Take QASM file/string as input and convert it to PennyLane circuit
    '''
    qasm_instructions = qasm_string.split('\n')
    instructions_without_metadata = [line for line in qasm_instructions[3:]]
    new_qasm = '\n'.join(instructions_without_metadata)
    gate_list = parse_qasm_file(new_qasm)
    angle_list = [instruction['angle'] for instruction in gate_list if instruction['angle'] is not None]
    if params is not None:
        cnt = 0
    for instruction in gate_list:
        if instruction['gate_name'] == 'x':
            qml.PauliX(wires=instruction['qubit_number'])
        elif instruction['gate_name'] == 'y':
            qml.PauliY(wires=instruction['qubit_number'])
        elif instruction['gate_name'] == 'z':
            qml.PauliZ(wires=instruction['qubit_number'])
        elif instruction['gate_name'] == 'h':
            qml.Hadamard(wires=instruction['qubit_number'])
        elif instruction['gate_name'] == 's':
            qml.S(wires=instruction['qubit_number'])
        elif instruction['gate_name'] == 't':
            qml.T(wires=instruction['qubit_number'])
        elif instruction['gate_name'] == 'id':
            continue
        elif instruction['gate_name'] == 'sx':
            qml.SX(wires=instruction['qubit_number'])
        elif instruction['gate_name'] == 'sxdg':
            qml.adjoint(qml.SX(wires=instruction['qubit_number']))
        elif instruction['gate_name'] == 'sdg':
            qml.adjoint(qml.S(wires=instruction['qubit_number']))
        elif instruction['gate_name'] == 'tdg':
            qml.adjoint(qml.T(wires=instruction['qubit_number']))
        elif instruction['gate_name'] == 'cx':
            qml.CNOT(wires=instruction['qubit_number'])
        elif instruction['gate_name'] == 'rx':
            if params is not None:
                qml.RX(params[cnt], wires=instruction['qubit_number'])
                cnt += 1
            else:
                qml.RX(instruction['angle'], wires=instruction['qubit_number'])
        elif instruction['gate_name'] == 'ry':
            if params is not None:
                qml.RY(params[cnt], wires=instruction['qubit_number'])
                cnt += 1
            else:
                qml.RY(instruction['angle'], wires=instruction['qubit_number'])
        elif instruction['gate_name'] == 'rz':
            if params is not None:
                qml.RZ(params[cnt], wires=instruction['qubit_number'])
                cnt += 1
            else:
                qml.RZ(instruction['angle'], wires=instruction['qubit_number'])


if __name__=="__main__":
    # Convert model to qasm
    model_qasm = get_trained_pqc_qasm()

    # Convert qasm to list of gates containing important metadata like gate name, angle, qubit number
    gate_list = parse_qasm_file(model_qasm)

    # Perform greedy optimization of rotation gates and obtain the reconstructed circuit
    qc_reconstructed = pqc_greedy_optimization(gate_list, etol=0.001)
    qc_reconstructed_transpiled = transpile(qc_reconstructed, basis_gates=['sx', 'cx', 'rz', 'id', 'x'], optimization_level=2)
    print(f"Reconstructed circuit data: Depth:{qc_reconstructed_transpiled.depth()}, Gate count: {qc_reconstructed_transpiled.count_ops()}")

    # Convert reconstructed circuit to qasm
    reconstructed_qasm = qc_reconstructed.qasm()
    new_gate_list = parse_qasm_file(reconstructed_qasm)
    new_param_list = [instruction['angle'] for instruction in new_gate_list if instruction['angle'] is not None]
    # Convert qasm to pennylane circuit
    @qml.qnode(dev)
    def pqc_circuit_iris_reconstructed(inputs, params):
        qml.AngleEmbedding(inputs, wires=range(4))
        add_pennylane_gates(reconstructed_qasm, params=params)
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]
    
    @qml.qnode(dev)
    def pqc_iris_strong_reconstructed(inputs, params):
        qml.AngleEmbedding(inputs, wires=range(4))
        add_pennylane_gates(reconstructed_qasm, params=params)
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    @qml.qnode(dev)
    def pqc_circuit_digits_reconstructed(inputs, params):
        qml.AmplitudeEmbedding(inputs, wires=range(6), normalize=True)
        add_pennylane_gates(reconstructed_qasm, params)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]
    
    @qml.qnode(dev)
    def pqc_digits_strong_reconstructed(inputs, params):
        qml.AmplitudeEmbedding(inputs, wires=range(6), normalize=True)
        add_pennylane_gates(reconstructed_qasm, params)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]
    
    # Define the TorchLayer
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            weight_shapes = {'params':(len(new_param_list),)}
            init_method = {'params':torch.tensor(new_param_list)}
            # Change according to dataset and ansatz used
            self.pqc = qml.qnn.TorchLayer(pqc_digits_strong_reconstructed, weight_shapes, init_method=init_method)

        def forward(self, x):
            return self.pqc(x)
    
    # Define the model and softmax function
    model = Model().to(device)
    soft_out = nn.Softmax(dim=1)

    # load iris dataset
    # iris = load_iris()
    # X = iris.data
    # y = iris.target

    # Load digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    indices = np.where((y == 0) | (y == 1))
    X = X[indices]
    y = y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform inferencing
    test_acc = 0
    for i, (inputs, labels) in enumerate(DataLoader(list(zip(X_test, y_test)), batch_size=16)):
        outputs = model(inputs.float())
        soft_outputs = soft_out(outputs)
        pred = torch.argmax(soft_outputs, dim=1)
        dist = torch.abs(labels - pred)
        test_acc += len(dist[dist==0])
    print("Reconstructed circuit test accuracy:", test_acc/len(X_test))

    assert len(new_param_list) != 0, "No rotation gates found in reconstructed circuit due to excessive optimization!"
    # Train the optimized model for 30% of original epochs (30% of 50 = 15)
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    extra_epochs = 20

    # Define the training loop
    print("Performing extra training on reconstructed circuit")
    for epoch in range(extra_epochs):
        train_acc = test_acc = 0    
        for i, (inputs, labels) in enumerate(DataLoader(list(zip(X_train, y_train)), batch_size=16)):
            optimizer.zero_grad()
            outputs = model(inputs.float())
            soft_outputs = soft_out(outputs)
            pred = torch.argmax(soft_outputs, dim=1)
            loss = loss_fn(outputs, labels.long())
            loss.backward()
            optimizer.step()
            dist = torch.abs(labels - pred)
            train_acc += len(dist[dist==0])

        for i, (inputs, labels) in enumerate(DataLoader(list(zip(X_test, y_test)), batch_size=16)):
            outputs = model(inputs.float())
            soft_outputs = soft_out(outputs)
            pred = torch.argmax(soft_outputs, dim=1)
            dist = torch.abs(labels - pred)
            test_acc += len(dist[dist==0])
        print("Epoch:", epoch+1,"Train accuracy:", train_acc/len(X_train) ,", Test accuracy:", test_acc/len(X_test))
    
    # Perform post training inferencing of reconstructed circuit
    print("Performing inferencing on reconstructed circuit")
    test_acc = 0
    for i, (inputs, labels) in enumerate(DataLoader(list(zip(X_test, y_test)), batch_size=16)):
        outputs = model(inputs.float())
        soft_outputs = soft_out(outputs)
        pred = torch.argmax(soft_outputs, dim=1)
        dist = torch.abs(labels - pred)
        test_acc += len(dist[dist==0])
    print("Reconstructed circuit test accuracy:", test_acc/len(X_test))

    



