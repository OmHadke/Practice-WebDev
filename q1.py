#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: lab2_multiQubitGate.ipynb
Conversion Date: 2025-11-27T04:11:36.540Z
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, Aer
from qiskit.visualization import plot_histogram
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector

print(qiskit.__version__)
print(Aer.backends())

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit

cr = ClassicalRegister(1,"cr")
qr = QuantumRegister(1,"qr")
qc = QuantumCircuit(qr,cr)

fig = qc.draw(output='mpl')
fig.savefig('q1_01_empty_circuit.png')
plt.close(fig)

# ====== PAULI X GATE ======

qr_x = QuantumRegister(1, 'qr_x')
cr_x = ClassicalRegister(1, 'cr_x')
qc_x = QuantumCircuit(qr_x)
qc_x.x(qr_x[0])
fig = qc_x.draw(output='mpl')
fig.savefig('q1_02_x_gate_circuit.png')
plt.close(fig)

backend = Aer.get_backend('statevector_simulator')
transpiled_qc_x = transpile(qc_x, backend)
result = backend.run(transpiled_qc_x).result().get_statevector(qc_x, decimals=3)
print("Result after implementing x-gate:",result)

fig = plot_bloch_multivector(result)
fig.savefig('q1_03_x_gate_bloch.png')
plt.close(fig)

backend = Aer.get_backend('unitary_simulator')
result = backend.run(transpiled_qc_x).result().get_unitary(qc_x, decimals=3)
print("Unitary matrix after implementing x-gate:\n",result)

cr_x = ClassicalRegister(1, 'classical register')
qr_x = QuantumRegister(1, 'quantum register')
qc_x = QuantumCircuit(qr_x, cr_x)
qc_x.x(qr_x[0])
qc_x.measure(qr_x, cr_x)
fig = qc_x.draw(output='mpl')
fig.savefig('q1_04_x_gate_measured.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
job = transpile(qc_x, backend)
result = backend.run(job).result()
counts = result.get_counts(qc_x)
print("Counts after applying X-gate and measuring the qubit:",counts)

fig = plot_histogram(counts)
fig.savefig('q1_05_x_gate_histogram.png')
plt.close(fig)

qc_x = QuantumCircuit(2,2,name='qc_x')
qc_x.x(0)
qc_x.barrier()
qc_x.measure([0,1],[0,1])
fig = qc_x.draw(output='mpl')
fig.savefig('q1_06_x_gate_2qubit_measured.png')
plt.close(fig)

qc_x = QuantumCircuit(2,name='qc_x')
qc_x.x(0)
qc_x.measure_all()
fig = qc_x.draw(output='mpl')
fig.savefig('q1_07_x_gate_2qubit_measure_all.png')
plt.close(fig)

qc_x = QuantumCircuit(2,2,name='qc_x')
qc_x.x(0)
qc_x.barrier()
qc_x.measure([0,1],[0,1])
fig = qc_x.draw(output='mpl')
fig.savefig('q1_08_x_gate_2qubit_measured2.png')
plt.close(fig)

qc_x = QuantumCircuit(2,name='qc_x')
qc_x.x(0)
qc_x.measure_all()
fig = qc_x.draw(output='mpl')
fig.savefig('q1_09_x_gate_2qubit_measure_all2.png')
plt.close(fig)

# ====== PAULI Y GATE ======

qr_y = QuantumRegister(1, 'qr_y')
cr_y = ClassicalRegister(1, 'cr_y')
qc_y = QuantumCircuit(qr_y)
qc_y.y(qr_y[0])
fig = qc_y.draw(output='mpl')
fig.savefig('q1_10_y_gate_circuit.png')
plt.close(fig)

backend = Aer.get_backend('statevector_simulator')
transpiled_qc_y = transpile(qc_y, backend)
result = backend.run(transpiled_qc_y).result().get_statevector(qc_y, decimals=3)
print("Result after implementing y-gate:\n",result)

fig = plot_bloch_multivector(result)
fig.savefig('q1_11_y_gate_bloch.png')
plt.close(fig)

backend = Aer.get_backend('unitary_simulator')
result = backend.run(transpiled_qc_y).result().get_unitary(qc_y, decimals=3)
print("Unitary matrix after implementing y-gate:\n",result)

qr_y = QuantumRegister(1, 'quantum register')
cr_y = ClassicalRegister(1, 'classical register')
qc_y = QuantumCircuit(qr_y, cr_y)
qc_y.y(qr_y[0])
qc_y.measure(qr_y, cr_y)
fig = qc_y.draw(output='mpl')
fig.savefig('q1_12_y_gate_measured.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
job = transpile(qc_y, backend)
result = backend.run(job).result()
counts = result.get_counts(qc_y)
print("Counts after applying Y-gate and measuring the qubit:",counts)

fig = plot_histogram(counts)
fig.savefig('q1_13_y_gate_histogram.png')
plt.close(fig)

qc_y = QuantumCircuit(2,2,name='qc_y')
qc_y.y(0)
qc_y.barrier()
qc_y.measure([0,1],[0,1])
fig = qc_y.draw(output='mpl')
fig.savefig('q1_14_y_gate_2qubit_measured.png')
plt.close(fig)

qc_y = QuantumCircuit(2,name='qc_y')
qc_y.y(0)
qc_y.measure_all()
fig = qc_y.draw(output='mpl')
fig.savefig('q1_15_y_gate_2qubit_measure_all.png')
plt.close(fig)

# ====== PAULI Z GATE ======

qc_z = QuantumCircuit(1,1,name='qc_z')
qc_z.x(0)
qc_z.z(0)
fig = qc_z.draw(output='mpl')
fig.savefig('q1_16_z_gate_circuit.png')
plt.close(fig)

backend = Aer.get_backend('statevector_simulator')
transpiled_qc_z = transpile(qc_z, backend)
result = backend.run(transpiled_qc_z).result().get_statevector(qc_z, decimals=3)
print("Result after implementing z-gate:\n",result)

fig = plot_bloch_multivector(result)
fig.savefig('q1_17_z_gate_bloch.png')
plt.close(fig)

backend = Aer.get_backend('unitary_simulator')
result = backend.run(transpiled_qc_z).result().get_unitary(qc_z, decimals=3)
print("Unitary matrix after implementing z-gate:\n",result)

qc_z = QuantumCircuit(1,1,name='qc_y')
qc_z.x(0)
qc_z.z(0)
qc_z.measure(0,0)
fig = qc_z.draw(output='mpl')
fig.savefig('q1_18_z_gate_measured.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
job = transpile(qc_z, backend)
result = backend.run(job).result()
counts = result.get_counts(qc_z)
print("Counts after applying Y-gate and measuring the qubit:",counts)

fig = plot_histogram(counts)
fig.savefig('q1_19_z_gate_histogram.png')
plt.close(fig)

qc_z = QuantumCircuit(2,2,name='qc_z')
qc_z.x(0)
qc_z.z(0)
qc_z.barrier()
qc_z.measure([0,1],[0,1])
fig = qc_z.draw(output='mpl')
fig.savefig('q1_20_z_gate_2qubit_measured.png')
plt.close(fig)

qc_z = QuantumCircuit(2,name='qc_z')
qc_z.x(0)
qc_z.z(0)
qc_z.measure_all()
fig = qc_z.draw(output='mpl')
fig.savefig('q1_21_z_gate_2qubit_measure_all.png')
plt.close(fig)

# ====== HADAMARD GATE ======

qc_h = QuantumCircuit(1,1,name='qc_h')
qc_h.h(0)
fig = qc_h.draw(output='mpl')
fig.savefig('q1_22_h_gate_circuit.png')
plt.close(fig)

backend = Aer.get_backend('statevector_simulator')
transpiled_qc_h = transpile(qc_h, backend)
result = backend.run(transpiled_qc_h).result().get_statevector(qc_h, decimals=3)
print("Result after implementing H-gate:\n",result)

fig = plot_bloch_multivector(result)
fig.savefig('q1_23_h_gate_bloch.png')
plt.close(fig)

backend = Aer.get_backend('unitary_simulator')
result = backend.run(transpiled_qc_h).result().get_unitary(qc_h, decimals=3)
print("Unitary matrix after implementing H-gate:\n",result)

qc_h = QuantumCircuit(1,1,name='qc_y')
qc_h.h(0)
qc_h.barrier()
qc_h.measure(0,0)
fig = qc_h.draw(output='mpl')
fig.savefig('q1_24_h_gate_measured.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
job = transpile(qc_h, backend)
result = backend.run(job, shots=1000).result()
counts = result.get_counts(qc_h)
print("Counts after applying H-gate and measuring the qubit:",counts)

fig = plot_histogram(counts)
fig.savefig('q1_25_h_gate_histogram.png')
plt.close(fig)

qc_h = QuantumCircuit(1,1,name='qc_y')
qc_h.x(0)
qc_h.h(0)
qc_h.barrier()
qc_h.measure(0,0)
fig = qc_h.draw(output='mpl')
fig.savefig('q1_26_xh_gate_measured.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
job = transpile(qc_h, backend)
result = backend.run(job,shots = 1000).result()
counts = result.get_counts(qc_h)
print("Counts after applying X-gate then H-gate and measuring the qubit:",counts)

fig = plot_histogram(counts)
fig.savefig('q1_27_xh_gate_histogram.png')
plt.close(fig)

# ====== 2 QUBITS ======

qc_h = QuantumCircuit(2,name='qc_z')
qc_h.x(0)
qc_h.h(0)
qc_h.x(1)
qc_h.h(1)
qc_h.measure_all()
fig = qc_h.draw(output='mpl')
fig.savefig('q1_28_2qubit_xh_circuit.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
job = transpile(qc_h, backend)
result = backend.run(job,shots = 1000).result()
counts = result.get_counts(qc_h)
print("Counts after applying X-gate then H-gate and measuring the qubit:",counts)

fig = plot_histogram(counts)
fig.savefig('q1_29_2qubit_xh_histogram.png')
plt.close(fig)

backend = Aer.get_backend('statevector_simulator')
transpiled_qc_h = transpile(qc_h, backend)
result = backend.run(transpiled_qc_h).result().get_statevector(qc_h, decimals=3)
print("Result after implementing H-gate:\n",result)

fig = plot_bloch_multivector(result)
fig.savefig('q1_30_2qubit_xh_bloch.png')
plt.close(fig)

# ====== 3 QUBITS ======

qc_h = QuantumCircuit(3,name='qc_z')
qc_h.x(0)
qc_h.h(0)
qc_h.z(1)
qc_h.h(1)
qc_h.y(2)
qc_h.h(2)
qc_h.measure_all()
fig = qc_h.draw(output='mpl')
fig.savefig('q1_31_3qubit_xzyhxhy_circuit.png')
plt.close(fig)

backend = Aer.get_backend('qasm_simulator')
job = transpile(qc_h, backend)
result = backend.run(job,shots = 1000).result()
counts = result.get_counts(qc_h)
print("Counts after applying X-gate then H-gate and measuring the qubit:",counts)

fig = plot_histogram(counts)
fig.savefig('q1_32_3qubit_xzyhxhy_histogram.png')
plt.close(fig)

backend = Aer.get_backend('statevector_simulator')
transpiled_qc_h = transpile(qc_h, backend)
result = backend.run(transpiled_qc_h).result().get_statevector(qc_h, decimals=3)
print("Result after implementing H-gate:\n",result)

fig = plot_bloch_multivector(result)
fig.savefig('q1_33_3qubit_xzyhxhy_bloch.png')
plt.close(fig)

print("\nAll visualizations saved successfully!")
