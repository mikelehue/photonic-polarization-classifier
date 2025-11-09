# Photonic Polarization Classifier

**Photonic polarization classifier with motorized waveplates (Zaber + Standa) and a Thorlabs PAX1000 polarimeter.**  
Implements a variational, *quantum-inspired clustering* algorithm based on the polarization of light — the photonic analogue of a single-qubit variational circuit.

This repository contains both the **numerical simulation** and the **experimental control code** used in the publication:

> **M. Varga, P. Bermejo, R. Pellicer-Guridi, R. Orús & G. Molina-Terriza (2024)**  
> *Quantum-inspired clustering with light*, **Scientific Reports 14, 21726**  
> [https://doi.org/10.1038/s41598-024-73053-z](https://doi.org/10.1038/s41598-024-73053-z)

---

## 🧠 Overview

This work demonstrates a **photonic simulation of a variational clustering algorithm** using polarization states of light.  
Inspired by quantum variational circuits, the system performs unsupervised classification of data points encoded on the Poincaré sphere.

- **Physical implementation:**  
  A diode laser (808 nm) and a sequence of **motorized waveplates** (Zaber + Standa) control the polarization state.  
  The resulting state is measured using a **Thorlabs PAX1000** polarimeter.

- **Numerical simulation:**  
  The same algorithm can be simulated in software, replacing the optical transformations by their Jones matrix equivalents.  
  This allows the exploration of optimization landscapes, initialization sensitivity, and cost-function behavior.

---

## 📂 Repository structure

photonic-polarization-classifier/  
│  
├── run_experiment.py          # Main experimental routine (hardware control + optimization loop)  
├── run_simulation.py          # Pure software simulation of the same algorithm  
│  
├── cost.py                    # Cost function definition for variational optimization  
├── labels_sphere.py           # Generation of cluster label states on the Bloch/Poincaré sphere  
├── cartesian_to_coeffs.py     # Conversion between (ψ, χ) coordinates and Jones vectors  
├── centroid_utils.py          # Helper functions for centroid computation  
├── distance_utils.py          # Distance metrics used in the cost function  
├── fibonacci_sphere.py        # Uniform point distribution over the sphere  
├── plot_results.py            # Visualization of cost landscapes and trajectories  
│  
├── pyximc.py                  # Low-level driver for Standa motor control  
├── calibration/  
│   └── cal_def_200x200_february_2024.txt   # Polarization calibration lookup table  
│  
└── README.md  

---

## ⚙️ Requirements

- Python 3.9+  
- numpy  
- scipy  
- matplotlib  
- zaber.serial  
- pyximc (included)  
- ctypes (standard library)  

Install dependencies:

pip install numpy scipy matplotlib zaber.serial  

---

## 🚀 Usage

### ▶ Simulation (software-only)

Run the clustering simulation:

python run_simulation.py  

This will:  
- Generate random Gaussian clusters on the Poincaré sphere  
- Apply virtual waveplate transformations  
- Optimize the variational angles via gradient descent  
- Display the optimization trajectories over the cost landscape  

### 🔬 Experiment (hardware)

Run the full experiment with Zaber and Standa stages and a Thorlabs PAX1000 polarimeter:

python run_experiment.py  

You will be prompted for a folder name where all experimental data (.npy) will be saved.  

---

## 📊 Results

Both the **simulation** and **experiment** reproduce the results presented in the paper, achieving **unsupervised clustering** of up to five Gaussian distributions on the polarization sphere.  
Experimental and numerical results show excellent agreement with minimal drift or noise dependence.  

---
