# -*- coding: utf-8 -*-

import os
import matplotlib
# Use a non-interactive backend so the script can run in headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# If a diagram image named `qft.png` exists (from the notebook), display it.
# This is a safe, script-friendly replacement for Colab's `IPython.display.Image`.
if os.path.exists("qft.png"):
    img = plt.imread("qft.png")
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

import qiskit
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator, Aer
from qiskit.visualization import plot_histogram
import matplotlib
from math import pi, sqrt # pi = 3.14 and square root operation
from qiskit.visualization import plot_state_city # Density Matrix plot
from qiskit.visualization import plot_state_qsphere # used for multi qubit visualisation
from qiskit.visualization import plot_bloch_multivector # Plotting the Bloch Sphere for Single Qubits
from qiskit.visualization import plot_histogram # 2D Histogram Plotting
import numpy as np # Numerical Python Library
from qiskit.quantum_info import Operator
from qiskit.circuit.library import ZGate

# Function to compute the QFT

def qft_rotations(circuit, n):
    if n == 0: # Exit function if circuit is empty
        return circuit
    n -= 1 # Indexes start from 0
    circuit.h(n) # Apply the H-gate to the most significant qubit
    for qubit in range(n):
        # For each less significant qubit, we need to do a
        # smaller-angled controlled rotation:
        circuit.cp(pi/2**(n-qubit), qubit, n)

    qft_rotations(circuit, n)

qc = QuantumCircuit(3)
qft_rotations(qc,3)
fig = qc.draw('mpl')
fig.savefig('qft_rotations_qc.png')
plt.close(fig)
plt.show()



# Function to SWAP the registers as seen in lecture
def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit



# Function to apply the QFT to the Quantum Circuit
def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

"""## QFT|000>"""

# QFT circuit for a 3 Qubit case at state |000>
qc = QuantumCircuit(3)
qft(qc,3)
fig = qc.draw('mpl')
fig.savefig('qft_000.png')
plt.close(fig)

"""## Visualzing the QFT results using Bloch Sphere"""

# Let's see the result

# To get the eigenvector you should use the statevector simulator in the core of the circuit (without measurements)
backend = Aer.get_backend('statevector_simulator')

transpiled_qc = transpile(qc, backend)

# Execute the circuit
result = backend.run(qc).result().get_statevector(qc, decimals=3)

# Printing the state
print("\nQuantum state is:",result)

# Plotting the Bloch Sphere
fig = plot_bloch_multivector(result)
fig.savefig('bloch_qft_000.png')
plt.close(fig)
plt.show()

"""The state received above is $QFT|000> = |+++>$

## QFT|101>
"""

# Circuit for a 3 Qubit QFT
qc = QuantumCircuit(3)

# Encode the state 5 - |101>
qc.x(0)
qc.x(2)

# Applying the QFT to the circuit with input as |101>
qft(qc,3)
fig = qc.draw('mpl')
fig.savefig('qft_101.png')
plt.close(fig)

# Let's see the result

# To get the eigenvector you should use the statevector simulator in the core of the circuit (without measurements)
backend = Aer.get_backend('statevector_simulator')

transpiled_qc = transpile(qc, backend)

# Execute the circuit

# Printing the state
print("\nQuantum state is:",result)

# Plotting the Bloch Sphere
fig = plot_bloch_multivector(result)
fig.savefig('bloch_qft_101.png')
plt.close(fig)

"""The state $|\widetilde{5}>$ which we have received in the above diagram is the QFT of $|5> = |101>$ and comparing this state with the $QFT|000> = |+++>$, it can be seen that qubit 0 has $\frac{5}{8}$ of full turn, qubit 1 has $\frac{1}{4}$ of full turn and qubit 2 has $\frac{1}{2} $ of full turn as seen in the theory lectures.

## Inverse Quantum Fourier Transform

We create the state $|\widetilde{5}>$ and run the QFT in reverse. Finally we verify that the output is the state $|5⟩ = |101>$ as expected.
"""

def inverse_qft(circuit, n):
    """Does the inverse QFT on the first n qubits in circuit"""
    # First we create a QFT circuit of the correct size:
    qft_circ = qft(QuantumCircuit(n), n)
    # Then we take the inverse of this circuit
    invqft_circ = qft_circ.inverse()
    # And add it to the first n qubits in our existing circuit
    circuit.append(invqft_circ, circuit.qubits[:n])
    return circuit.decompose() # .decompose() allows us to see the individual gates

# Creating the QFT|101> state manually
nqubits = 3
number = 5
qc = QuantumCircuit(nqubits)
for qubit in range(nqubits):
    qc.h(qubit)
qc.p(number*pi/4,0)
qc.p(number*pi/2,1)
qc.p(number*pi,2)

# Let's see the result of QFT|101>
fig = qc.draw('mpl')
fig.savefig('qft_manual_101.png')
plt.close(fig)
# Let's see the result of QFT|101>

# Let's see the result of QFT|101>

# To get the eigenvector you should use the statevector simulator in the core of the circuit (without measurements)
backend = Aer.get_backend('statevector_simulator')

transpiled_qc = transpile(qc, backend)

# Execute the circuit
result = backend.run(qc).result().get_statevector(qc, decimals=3)

# Printing the state
print("\nQuantum state is:",result)

# Plotting the Bloch Sphere
fig = plot_bloch_multivector(result)
fig.savefig('bloch_manual_101.png')
plt.close(fig)

# Calling the inverse QFT Function
qc = inverse_qft(qc, nqubits)
qc.measure_all()
fig = qc.draw('mpl')
fig.savefig('inverse_qft_draw.png')
plt.close(fig)

# Use Aer's qasm_simulator
backend = Aer.get_backend('qasm_simulator')

transpiled_qc = transpile(qc, backend)

# Execute the circuit on the qasm simulator
job = backend.run(qc, shots=1000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(qc)
print("\nTotal counts are:",counts)

# Plot the histogram
fig = plot_histogram(counts)
fig.savefig('q8_counts_histogram.png')
plt.close(fig)

"""As expected, we are able to get the state |101> which is number 5 in decimal."""
