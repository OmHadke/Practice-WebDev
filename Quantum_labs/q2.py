#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: lab3_analysingEntanglement.ipynb
Conversion Date: 2025-11-27T04:04:44.786Z
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, Aer
from qiskit.visualization import plot_histogram
from math import pi, sqrt
from qiskit.visualization import plot_state_city
from qiskit.visualization import plot_state_qsphere
from qiskit.visualization import plot_bloch_multivector
import numpy as np

# ====== CNOT/CX GATE SECTION ======

# CX-gate on |11> = |10> (|01> according to Qiskit ordering)
qc_cx = QuantumCircuit(2,name="qc")
qc_cx.x(0)
qc_cx.x(1)
qc_cx.cx(0,1)
fig = qc_cx.draw('mpl')
fig.savefig('q2_01_cx_gate_circuit.png')
plt.close(fig)

# Statevector simulation
simulator_state = Aer.get_backend('statevector_simulator')
job_state = transpile(qc_cx, simulator_state)
result_state = simulator_state.run(job_state).result()
psi = result_state.get_statevector(qc_cx)
print("\nQuantum state is:",psi)

fig = plot_state_city(psi)
fig.savefig('q2_02_cx_state_city.png')
plt.close(fig)

fig = plot_state_qsphere(psi)
fig.savefig('q2_03_cx_qsphere.png')
plt.close(fig)

# Unitary simulation
backend = Aer.get_backend('unitary_simulator')
transpiled_qc = transpile(qc_cx, backend)
job = backend.run(transpiled_qc)
result = job.result()
unitary_matrix = result.get_unitary()
print("Unitary matrix of the circuit:")
print(np.round(unitary_matrix.data, 3))

# Measurement simulation
qc_cx = QuantumCircuit(2,2,name="qc")
qc_cx.x(0)
qc_cx.x(1)
qc_cx.cx(0,1)
qc_cx.measure([0,1],[0,1])
fig = qc_cx.draw('mpl')
fig.savefig('q2_04_cx_measured_circuit.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
transpiled_qc = transpile(qc_cx, backend)
job = backend.run(transpiled_qc, shots=1000)
result = job.result()
counts = result.get_counts()
print("\nTotal counts are:", counts)

fig = plot_histogram(counts)
fig.savefig('q2_05_cx_histogram.png')
plt.close(fig)

# H-CZ-H alternative
qc_cx = QuantumCircuit(2,2,name="qc")
qc_cx.x(0)
qc_cx.x(1)
qc_cx.barrier()
qc_cx.h(1)
qc_cx.cz(0,1)
qc_cx.h(1)
qc_cx.barrier()
qc_cx.measure([0,1],[0,1])
fig = qc_cx.draw('mpl')
fig.savefig('q2_06_cx_hczh_circuit.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
transpiled_qc = transpile(qc_cx, backend)
job = backend.run(transpiled_qc, shots=1000)
result = job.result()
counts = result.get_counts()
print("\nTotal counts are:", counts)

fig = plot_histogram(counts)
fig.savefig('q2_07_cx_hczh_histogram.png')
plt.close(fig)

# ====== CZ GATE SECTION ======

qc_cz = QuantumCircuit(2,name="qc")
qc_cz.x(0)
qc_cz.x(1)
qc_cz.cz(0,1)
fig = qc_cz.draw('mpl')
fig.savefig('q2_08_cz_circuit.png')
plt.close(fig)

simulator_state = Aer.get_backend('statevector_simulator')
transpiled_qc = transpile(qc_cz, simulator_state)
job_state = simulator_state.run(transpiled_qc)
result_state = job_state.result()
psi = result_state.get_statevector()
print("\nQuantum state is:", psi)

fig = plot_state_city(psi)
fig.savefig('q2_09_cz_state_city.png')
plt.close(fig)

fig = plot_state_qsphere(psi)
fig.savefig('q2_10_cz_qsphere.png')
plt.close(fig)

backend = Aer.get_backend('unitary_simulator')
transpiled_qc = transpile(qc_cz, backend)
job = backend.run(transpiled_qc)
result = job.result()
unitary_matrix = result.get_unitary()
print("Unitary matrix of the circuit:")
print(np.round(unitary_matrix.data, 3))

qc_cz = QuantumCircuit(2,2,name="qc")
qc_cz.x(0)
qc_cz.x(1)
qc_cz.cz(0,1)
qc_cz.measure([0,1],[0,1])
fig = qc_cz.draw('mpl')
fig.savefig('q2_11_cz_measured_circuit.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
transpiled_qc = transpile(qc_cz, backend)
job = backend.run(transpiled_qc, shots=1000)
result = job.result()
counts = result.get_counts()
print("\nTotal counts are:", counts)

fig = plot_histogram(counts)
fig.savefig('q2_12_cz_histogram.png')
plt.close(fig)

# H-CX-H alternative
qc_cz = QuantumCircuit(2,2,name="qc")
qc_cz.x(0)
qc_cz.x(1)
qc_cz.barrier()
qc_cz.h(1)
qc_cz.cx(0,1)
qc_cz.h(1)
qc_cz.barrier()
qc_cz.measure([0,1],[0,1])
fig = qc_cz.draw('mpl')
fig.savefig('q2_13_cz_hcxh_circuit.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
transpiled_qc = transpile(qc_cz, backend)
job = backend.run(transpiled_qc, shots=1000)
result = job.result()
counts = result.get_counts()
print("\nTotal counts are:", counts)

fig = plot_histogram(counts)
fig.savefig('q2_14_cz_hcxh_histogram.png')
plt.close(fig)

# ====== CH GATE SECTION ======

qc_ch = QuantumCircuit(2,name="qc")
qc_ch.x(0)
qc_ch.x(1)
qc_ch.ch(0,1)
fig = qc_ch.draw('mpl')
fig.savefig('q2_15_ch_circuit.png')
plt.close(fig)

simulator_state = Aer.get_backend('statevector_simulator')
transpiled_qc = transpile(qc_ch, simulator_state)
job_state = simulator_state.run(transpiled_qc)
result_state = job_state.result()
psi = result_state.get_statevector()
print("\nQuantum state is:", psi)

fig = plot_state_city(psi)
fig.savefig('q2_16_ch_state_city.png')
plt.close(fig)

fig = plot_state_qsphere(psi)
fig.savefig('q2_17_ch_qsphere.png')
plt.close(fig)

backend = Aer.get_backend('unitary_simulator')
transpiled_qc = transpile(qc_ch, backend)
job = backend.run(transpiled_qc)
result = job.result()
unitary_matrix = result.get_unitary()
print("Unitary matrix of the circuit:")
print(np.round(unitary_matrix.data, 3))

qc_ch = QuantumCircuit(2,2,name="qc")
qc_ch.x(0)
qc_ch.x(1)
qc_ch.ch(0,1)
qc_ch.measure([0,1],[0,1])
fig = qc_ch.draw('mpl')
fig.savefig('q2_18_ch_measured_circuit.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
transpiled_qc = transpile(qc_ch, backend)
job = backend.run(transpiled_qc, shots=1000)
result = job.result()
counts = result.get_counts()
print("\nTotal counts are:", counts)

fig = plot_histogram(counts)
fig.savefig('q2_19_ch_histogram.png')
plt.close(fig)

# Ry decomposition
qc_ch = QuantumCircuit(2,2,name="qc")
qc_ch.x(0)
qc_ch.x(1)
qc_ch.barrier()
qc_ch.ry(pi/4,1)
qc_ch.cx(0,1)
qc_ch.ry(-pi/4,1)
qc_ch.barrier()
qc_ch.measure([0,1],[0,1])
fig = qc_ch.draw('mpl')
fig.savefig('q2_20_ch_ry_decomposition_circuit.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
transpiled_qc = transpile(qc_ch, backend)
job = backend.run(transpiled_qc, shots=1000)
result = job.result()
counts = result.get_counts()
print("\nTotal counts are:", counts)

fig = plot_histogram(counts)
fig.savefig('q2_21_ch_ry_decomposition_histogram.png')
plt.close(fig)

# ====== SWAP GATE SECTION ======

qc_sw = QuantumCircuit(2,name="qc")
qc_sw.x(1)
qc_sw.swap(0,1)
fig = qc_sw.draw(output='mpl')
fig.savefig('q2_22_swap_circuit.png')
plt.close(fig)

simulator_state = Aer.get_backend('statevector_simulator')
transpiled_qc = transpile(qc_sw, simulator_state)
job_state = simulator_state.run(transpiled_qc)
result_state = job_state.result()
psi = result_state.get_statevector()
print("\nQuantum state is:", psi)

fig = plot_state_city(psi)
fig.savefig('q2_23_swap_state_city.png')
plt.close(fig)

fig = plot_state_qsphere(psi)
fig.savefig('q2_24_swap_qsphere.png')
plt.close(fig)

backend = Aer.get_backend('unitary_simulator')
transpiled_qc = transpile(qc_sw, backend)
job = backend.run(transpiled_qc)
result = job.result()
unitary_matrix = result.get_unitary()
print("Unitary matrix of the circuit:")
print(np.round(unitary_matrix.data, 3))

qc_sw = QuantumCircuit(2,2,name="qc")
qc_sw.x(1)
qc_sw.swap(0,1)
qc_sw.measure([0,1],[0,1])
fig = qc_sw.draw(output='mpl')
fig.savefig('q2_25_swap_measured_circuit.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
transpiled_qc = transpile(qc_sw, backend)
job = backend.run(transpiled_qc, shots=1000)
result = job.result()
counts = result.get_counts()
print("\nTotal counts are:", counts)

fig = plot_histogram(counts)
fig.savefig('q2_26_swap_histogram.png')
plt.close(fig)

print("\nAll visualizations saved successfully!")
