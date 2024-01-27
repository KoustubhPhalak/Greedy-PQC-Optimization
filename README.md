# Greedy PQC Optimization

Novel methodology proposed to optimize Parametric Quantum Circuits (PQC) using greedy algorithm, where parametric gates are represented as set of approximate non-parametric gates. PyTorch + Pennylane implementation of xxx

## Important files
Primarily, this repository contains 3 important files:

1.```qnn_train.py```: contains code to train a PQC on Iris/Digits dataset using Basic Entangler Layer/ Strongly Entangling Layer as main ansatz

2. ```transformation_matrix_approximation.py```: contains the greedy algorithm code to greedily optimize single qubit gates of qnn based on the Hilbert-Schmidt (HS) distance metric $d = 1 - \frac{Tr(V^{\dagger}U)}{dim(V)}$, where $U$ is the unitary transformation matrix of the original gate and $V$ is the unitary transformation matrix of the greedily optimized set of non-parametric gates

3. ```qnn_transformation.py```: code to perform greedy optimization of trained qnn circuit

### Extra files
These files are initial experimental files and are not the main files to be used. However if the viewers can view them for experimental purposes if it helps them in their analysis

1. ```fidelity_approximation.py```: uses fidelity as optimization metric instead of HS distance. Not really a reliable metric as fidelity is related to the quantum state as compared to quantum gates

2. ```test_transformation.ipynb```: contains random test code used for implementing main codes.

## How to use? 
