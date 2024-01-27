'''
Contains code to train PQC on Iris and Digits dataset 
using basic and strongly entangling layers
'''

import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define model save path
model_save_path = "model_params_digits_strong.pth"
batch_size = 16

# Load the iris dataset
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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size)
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size)

# Define the quantum device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# For iris, 8 qubits, For digits, 10 qubits
qubits = 10
dev = qml.device("lightning.qubit", wires=qubits)

# Define the PQC circuit
@qml.qnode(dev)
def pqc_circuit_iris(inputs, params):
    qml.templates.AngleEmbedding(inputs, wires=range(4))
    qml.templates.BasicEntanglerLayers(params, wires=range(qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

@qml.qnode(dev)
def pqc_iris_strong(inputs, params):
    qml.templates.AngleEmbedding(inputs, wires=range(4))
    qml.templates.StronglyEntanglingLayers(params, wires=range(qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

@qml.qnode(dev)
def pqc_circuit_digits(inputs, params):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(6), normalize=True)
    qml.templates.BasicEntanglerLayers(params, wires=range(qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

@qml.qnode(dev)
def pqc_digits_strong(inputs, params):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(6), normalize=True)
    qml.templates.StronglyEntanglingLayers(params, wires=range(qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

# Define the TorchLayer
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"params": (5, qubits)}
        weight_shapes_strong = {"params": (5, qubits, 3)}
        # Change according to dataset and ansatz used
        self.pqc = qml.qnn.TorchLayer(pqc_digits_strong, weight_shapes_strong, init_method=torch.nn.init.normal_)

    def forward(self, x):
        return self.pqc(x)


# Define the model, optimizer and loss function
model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
soft_out = nn.Softmax(dim=1)
loss_fn = nn.CrossEntropyLoss()

# Define the training loop
for epoch in range(50):
    train_acc = test_acc = 0    
    for i, (inputs, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        outputs = model(inputs.float())
        soft_outputs = soft_out(outputs)
        pred = torch.argmax(soft_outputs, dim=1)
        loss = loss_fn(soft_outputs, labels.long())
        loss.backward()
        optimizer.step()
        dist = torch.abs(labels - pred)
        train_acc += len(dist[dist==0])
    for i, (inputs, labels) in enumerate(tqdm(test_loader)):
        outputs = model(inputs.float())
        soft_outputs = soft_out(outputs)
        pred = torch.argmax(soft_outputs, dim=1)
        dist = torch.abs(labels - pred)
        test_acc += len(dist[dist==0])
    print(f"Epoch {epoch+1}: Loss = {loss.item()}, Train Acc = {train_acc/len(X_train)}, Test Accuracy = {test_acc/len(X_test)}")

# # Save the parameters of the model
torch.save(model.state_dict(), model_save_path)

# Code for inferencing. Comment above loop and torch.save() while doing this
model.load_state_dict(torch.load(model_save_path))
test_acc = 0
for i, (inputs, labels) in enumerate(DataLoader(list(zip(X_test, y_test)), batch_size=16)):
    outputs = model(inputs.float())
    soft_outputs = soft_out(outputs)
    pred = torch.argmax(soft_outputs, dim=1)
    dist = torch.abs(labels - pred)
    test_acc += len(dist[dist==0])
print("Test accuracy:", test_acc/len(X_test))



