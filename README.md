# Greedy PQC Optimization

Novel methodology proposed to optimize Parametric Quantum Circuits (PQC) using greedy algorithm, where parametric gates are represented as set of approximate non-parametric gates. PyTorch + Pennylane implementation of *"Non-parametric Greedy Optimization of Parametric Quantum Circuits"*

## Important files
Primarily, this repository contains 3 important files:

1. ```qnn_train.py```: contains code to train a PQC on Iris/Digits dataset using Basic Entangler Layer (BEL)/ Strongly Entangling Layer (SEL) as main ansatz

2. ```transformation_matrix_approximation.py```: contains the greedy algorithm code to greedily optimize single qubit gates of QNN based on the Hilbert-Schmidt (HS) distance metric $d = 1 - \frac{Tr(V^{\dagger}U)}{dim(V)}$, where $U$ is the unitary transformation matrix of the original gate and $V$ is the unitary transformation matrix of the greedily optimized set of non-parametric gates

3. ```qnn_transformation.py```: code to perform greedy optimization of trained QNN circuit

### Extra files
These files are initial experimental files and are not the main files to be used. However if the viewers can view them for experimental purposes if it helps them in their analysis

1. ```fidelity_approximation.py```: uses fidelity as optimization metric instead of HS distance. Not really a reliable metric as fidelity is related to the quantum state as compared to quantum gates

2. ```test_transformation.ipynb```: contains random test code used for implementing main codes.

## How to use? 
Generally, it involves two steps: 1. train the model, and 2. perform optimization on trained model

1. ```qnn_train.py``` will train the model using desired dataset (Iris or Digits, lines ```19-30```) and desired ansatz (BEL or SEL, lines ```44-66```, line ```75```). Note that we use **8 qubits** for Iris dataset and **10 qubits** for Digits dataset. Furthermore, we peform classification of only 0-1 digits on Digits dataset and all 3 classes on Iris dataset.

2. ```qnn_transformation.py``` uses the algorithm implemented in ```transformation_matrix_approximation.py``` to optimize and reconstruct trained QNN. Once again, users can change the dataset (line ```24```, lines ```266-277```) and the ansatz (**import**: lines ```28-50```, line ```58```, **reconstruction**: lines ```225-247```, line ```256```). Make sure to choose the desired model path ( line ```65```)

## Python Library versions
The following are all the relevant library versions for all python libraries used:

```
python==3.8.5
torch==2.0.1
pennylane==0.33.1
qiskit==0.45.1
scikit-learn==1.3.2
numpy==1.22.0
tqdm==4.62.3
```

These libraries are not necessarily required to be at their respective above mentioned versions, however the code has been tried and tested successfully to work at these versions.