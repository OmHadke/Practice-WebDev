# Quantum Lab Files (q1-q9) - Installation & Setup Guide

## Overview
All 9 quantum computing lab files (q1.py through q9.py) are now fully converted from Jupyter notebooks to standalone Python scripts. They use Qiskit for quantum simulations and Matplotlib for visualization with headless backend support.

## Required Packages to Install

### Core Quantum Computing
- **qiskit** (2.2.3+) - Quantum computing framework
- **qiskit-aer** (0.17.2+) - High-performance quantum simulators

### Visualization & Graphics
- **matplotlib** (3.10.7+) - Plotting library (using Agg backend for headless mode)
- **seaborn** (0.13.2+) - Statistical data visualization (required for Qiskit's plot_state_qsphere)
- **pillow** (12.0.0+) - Image processing library (PIL)

### Scientific Computing
- **numpy** (2.3.5+) - Numerical computing library
- **scipy** (1.16.3+) - Scientific computing utilities

### LaTeX Support (for circuit diagram annotations)
- **pylatexenc** (2.10+) - LaTeX encoding utilities

### Additional Dependencies
- rustworkx (0.17.1+) - Graph library for Qiskit
- typing-extensions, dill, psutil, stevedore - Support libraries

---

## Installation Instructions for Local VS Code

### Option 1: Using the Existing requirements.txt (Recommended)

```bash
# Navigate to project directory
cd /path/to/Practice-WebDev

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### Option 2: Install Individual Packages (Minimal Setup)

```bash
pip install qiskit qiskit-aer matplotlib seaborn numpy scipy pylatexenc pillow
```

### Option 3: Install Highest-Priority Packages First

```bash
# Essential for all labs
pip install qiskit qiskit-aer matplotlib numpy

# Required for advanced visualizations (q2-q9)
pip install seaborn pylatexenc pillow scipy
```

---

## Verification - Test Installation

Run this Python snippet to verify all dependencies are installed:

```python
import sys
print(f"Python: {sys.version}")

import qiskit
print(f"✓ Qiskit: {qiskit.__version__}")

import qiskit_aer
print(f"✓ Qiskit-Aer: {qiskit_aer.__version__}")

import matplotlib
print(f"✓ Matplotlib: {matplotlib.__version__}")

import numpy
print(f"✓ NumPy: {numpy.__version__}")

import seaborn
print(f"✓ Seaborn: {seaborn.__version__}")

import pylatexenc
print(f"✓ PyLatexEnc: {pylatexenc.__version__}")

from PIL import Image
print(f"✓ Pillow: {Image.__version__}")

print("\n✓ All dependencies installed successfully!")
```

---

## File Summary & What Each Lab Requires

| File | Topic | Key Dependencies | Status |
|------|-------|------------------|--------|
| **q1.py** | Single-Qubit Gates (X, Y, Z, H) | qiskit, matplotlib, numpy | ✅ Fixed |
| **q2.py** | Multi-Qubit Gates (CX, CZ, CH, SWAP) | qiskit, matplotlib, seaborn, numpy | ✅ Fixed |
| **q3.py** | Entanglement & Bell States | qiskit, matplotlib, seaborn, numpy | ✅ Fixed |
| **q4.py** | Superdense Coding Protocol | qiskit, matplotlib, seaborn, numpy | ✅ Fixed |
| **q5.py** | Deutsch Algorithm | qiskit, matplotlib, numpy | ✅ Fixed |
| **q6.py** | Deutsch-Jozsa Algorithm | qiskit, matplotlib, numpy | ✅ Fixed |
| **q7.py** | Grover's Search Algorithm | qiskit, matplotlib, numpy | ✅ Fixed |
| **q8.py** | Quantum Fourier Transform (QFT) | qiskit, matplotlib, seaborn, numpy | ✅ Fixed |
| **q9.py** | Quantum Phase Estimation (QPE) | qiskit, matplotlib, seaborn, numpy | ✅ Ready |

---

## Running the Files

After installation, run any lab file:

```bash
python q1.py
python q2.py
python q3.py
# ... etc
```

### Output
Each script generates PNG visualizations (e.g., `q1_01_empty_circuit.png`, `q2_03_cx_qsphere.png`, etc.) in the same directory.

---

## VS Code Setup (Optional but Recommended)

### 1. Select Python Interpreter
- Press `Ctrl+Shift+P` → "Python: Select Interpreter"
- Choose your virtual environment (`./venv/bin/python` or your local environment)

### 2. Install VS Code Extensions
- **Python** (by Microsoft)
- **Pylance** (by Microsoft) - For type checking and IntelliSense
- **Jupyter** (by Microsoft) - For notebook support (optional)

### 3. Configure Settings
In `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "ms-python.python"
    }
}
```

### 4. Run with Play Button
- Click the "Run" button (▶️) in the top-right of any .py file
- Or use terminal: `python filename.py`

---

## Troubleshooting

### ImportError: No module named 'qiskit'
```bash
pip install qiskit qiskit-aer
```

### ImportError: No module named 'seaborn'
```bash
pip install seaborn
```

### matplotlib backend warning
All scripts use `matplotlib.use('Agg')` for headless mode - no action needed.

### PNG files not generating
- Check write permissions in the directory
- Verify matplotlib is installed: `pip install matplotlib --upgrade`
- Ensure no errors in output (look for exceptions)

---

## Complete Dependency List (with versions used in workspace)

```
contourpy==1.3.3
cycler==0.12.1
dill==0.4.0
fonttools==4.60.1
kiwisolver==1.4.9
matplotlib==3.10.7
numpy==2.3.5
packaging==25.0
pillow==12.0.0
psutil==7.1.3
pylatexenc==2.10
pyparsing==3.2.5
python-dateutil==2.9.0.post0
qiskit==2.2.3
qiskit-aer==0.17.2
rustworkx==0.17.1
scipy==1.16.3
six==1.17.0
stevedore==5.6.0
typing_extensions==4.15.0
```

---

## Minimal Installation (if bandwidth/space is a concern)

For basic functionality without visualization enhancements:

```bash
pip install qiskit qiskit-aer matplotlib numpy
```

**Note:** This may cause seaborn-related warnings in q2-q4, q8-q9, but scripts will still execute.

---

## Summary

✅ **All 9 quantum lab files are production-ready**
✅ **All dependencies are documented**
✅ **Complete setup instructions provided**
✅ **Scripts generate PNG visualizations** (no interactive display required)
✅ **Works in headless environments** (perfect for VS Code, SSH, containers)

