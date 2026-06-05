# Physics-Informed Neural Networks for Inverse Modelling of Endocrine Dynamics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
## Example Result

![Parameter Discovery](parameter_discovery.png)

![Parameter Convergence](parameter_convergence.png)

## 📌 Project Overview

This repository demonstrates the use of **Physics-Informed Neural Networks (PINNs)** for solving an inverse problem in computational endocrinology.

Rather than merely reconstructing physiological trajectories, the objective is to **recover hidden physiological parameters directly from sparse and noisy clinical observations**.

The project illustrates how Scientific Machine Learning (SciML) can combine mechanistic physiological knowledge with neural networks to estimate latent biological quantities that are not directly observable.

This approach is particularly relevant for digital health, computational biology, personalized medicine, and endocrine system modelling.

---

## 🧬 Scientific Motivation

In many biomedical applications, clinicians observe measurements such as glucose concentrations but cannot directly measure important physiological parameters, including:

* Glucose clearance rates
* Insulin sensitivity
* Hormone degradation constants
* Secretion dynamics

Traditional machine learning predicts observed variables.

Scientific Machine Learning aims to answer a deeper question:

> Which physiological parameters generated the observed data?

This process is known as **inverse modelling** or **parameter discovery**.

---

## 📖 Mathematical Model

The endocrine dynamics are constrained by a simplified physiological model:

```math
\frac{dG}{dt}
=
-p_1(G-G_b)
-
XG
```

where:

| Symbol | Description                 |
| ------ | --------------------------- |
| $G(t)$ | Blood glucose concentration |
| $X(t)$ | Latent insulin-action state |
| $G_b$  | Basal glucose concentration |
| $p_1$  | Glucose clearance parameter |

The parameter $p_1$ is treated as **unknown** and learned directly from data.

---

## 🎯 Inverse Modelling Objective

Unlike conventional PINN trajectory reconstruction, the goal is to estimate hidden physiological parameters.

### Known

```text
Sparse glucose observations
Basal glucose level
Physiological differential equation
```

### Unknown

```text
Glucose clearance parameter (p1)
Latent insulin-action state X(t)
Continuous glucose trajectory G(t)
```

The PINN simultaneously learns:

```text
G(t)
X(t)
p1
```

while satisfying both the observed data and the governing physiological equation.

---

## ⚙️ Physics-Informed Loss Function

The optimization objective combines data consistency and physiological realism.

### Data Loss

The model minimizes reconstruction error:

```math
\mathcal{L}_{data}
=
MSE
(
G_{pred},
G_{obs}
)
```

---

### Physics Loss

Automatic differentiation computes temporal derivatives and enforces compliance with the physiological model:

```math
\mathcal{L}_{physics}
=
MSE
(
ODE\ Residual
)
```

where

```math
Residual
=
\frac{dG}{dt}
+
p_1(G-G_b)
+
XG
```

---

### Total Loss

```math
\mathcal{L}_{total}
=
\mathcal{L}_{data}
+
\lambda
\mathcal{L}_{physics}
```

where

$\lambda$ controls the balance between:

* Data fidelity
* Physiological consistency

---

## 🏗️ Model Architecture

The PINN consists of a fully connected neural network:

```text
Input:
    Time (t)

Hidden Layer:
    64 neurons
    Tanh activation

Hidden Layer:
    64 neurons
    Tanh activation

Output:
    G(t)
    X(t)
```

Additionally:

```text
p1
```

is implemented as a trainable parameter and optimized jointly with the network weights.

---

## 📊 Example Workflow

```text
Synthetic Patient Data
           ↓
Sparse Glucose Measurements
           ↓
Physics-Informed Neural Network
           ↓
Trajectory Reconstruction
           ↓
Parameter Discovery
           ↓
Estimated Physiological Parameter
```

---

## 📈 Example Outputs

The project produces:

### 1. Glucose Trajectory Reconstruction

Shows agreement between:

* Clinical observations
* PINN reconstruction
* Physiological dynamics

---

### 2. Parameter Convergence

Tracks the learned parameter through training:

```text
True p1 = 0.025

Estimated p1 ≈ 0.025
```

demonstrating successful recovery of the hidden physiological parameter.

---

## 🛠️ Repository Structure

```text
├── main.py
├── parameter_discovery.png
├── parameter_convergence.png
├── README.md
└── requirements.txt
```

### Files

| File                      | Description                    |
| ------------------------- | ------------------------------ |
| main.py                   | Complete PINN implementation   |
| parameter_discovery.png   | Glucose reconstruction results |
| parameter_convergence.png | Parameter learning curve       |
| README.md                 | Project documentation          |
| requirements.txt          | Dependencies                   |

---

## ⚡ Installation

Install dependencies:

```bash
pip install torch numpy matplotlib
```

---

## ▶️ Running the Project

Execute:

```bash
python main.py
```

The script will:

1. Generate synthetic endocrine data.
2. Train a Physics-Informed Neural Network.
3. Estimate a hidden physiological parameter.
4. Reconstruct the glucose trajectory.
5. Visualize parameter convergence.

---

## 🔬 Scientific Machine Learning Concepts Demonstrated

This project demonstrates:

* Physics-Informed Neural Networks (PINNs)
* Scientific Machine Learning (SciML)
* Inverse Modelling
* Parameter Discovery
* System Identification
* Automatic Differentiation
* ODE-Constrained Learning
* Computational Endocrinology
* Mechanistic–Data-Driven Modelling
* Interpretable Machine Learning

---

## 🚀 Future Extensions

Potential research extensions include:

### Full Bergman Minimal Model

Estimate:

* Insulin sensitivity
* Glucose effectiveness
* Hormone dynamics

from clinical measurements.

---

### Personalized Digital Twins

Learn patient-specific physiological parameters and forecast future endocrine behavior.

---

### Bayesian PINNs

Quantify uncertainty in parameter estimates and predictions.

---

### Real Clinical Data

Apply the framework to:

* Continuous Glucose Monitoring (CGM)
* Oral Glucose Tolerance Tests (OGTT)
* Diabetes monitoring datasets

---

### Neural ODE Comparisons

Compare:

* Classical ODE fitting
* Neural ODEs
* PINNs

for endocrine system identification.

---

## 📚 Research Areas

This project sits at the intersection of:

* Scientific Machine Learning
* Computational Biology
* Computational Endocrinology
* Digital Health
* Dynamical Systems
* Inverse Problems
* Deep Learning
* Mathematical Modelling

---

## 📄 License

Distributed under the MIT License.

See `LICENSE` for more information.
