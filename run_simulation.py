# -*- coding: utf-8 -*-
"""
Run a *pure simulation* of the photonic polarization classifier.

- Generates synthetic clusters of (psi, chi) points on the Poincaré sphere.
- Converts them to Jones vectors.
- Applies a stack of virtual waveplates (quarter/half) parameterized by `params`.
- Uses a finite-difference gradient step to reduce a custom cost (see cost.py).
- Plots the optimization trajectories over a background cost map image.

Notes:
- This file **does not** talk to hardware. For the real experiment, use
  `run_experiment.py`.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from cost import cost_function
from labels_sphere import state_labels  # renamed from Labels_sphere

# Small aliases used in Jones/Stokes helpers
from numpy import cos as cos
from numpy import sin as sin
from numpy import pi as pi


# ---------------------------- Conversions ------------------------------------
def Stokes2Jones(psi, chi):
    """
    Convert Stokes angles (psi, chi) to a 2D Jones state [c1, c2].

    psi: azimuth (rad), chi: ellipticity (rad)
    Returns a length-2 complex vector normalized to 1.
    """
    # Stokes on the Poincaré sphere
    Q = cos(2 * psi) * cos(2 * chi)
    U = sin(2 * psi) * cos(2 * chi)
    V = sin(2 * chi)

    # Map to unit 3-vector (x,y,z)
    x, y, z = 2 * Q, 2 * U, 2 * V
    nrm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / nrm, y / nrm, z / nrm

    # Spherical angles on the Bloch/Poincaré sphere
    phi = np.arctan2(y, x)
    theta = np.arccos(z)

    # Jones coefficients of the corresponding pure state
    return np.array([cos(theta / 2), np.exp(1j * phi) * sin(theta / 2)], dtype=complex)


# --------------------------- Output directory --------------------------------
# (Kept simple; simulation data is small)
FOLDER = "data"
os.makedirs(FOLDER, exist_ok=True)


# ------------------------------ Classifier utils -----------------------------
def test(quantum_states, labels):
    """
    Compute |<state|label>|^2, returning (max_fidelities, argmax_indices)
    along the label axis.
    """
    fidelities = np.abs(np.matmul(quantum_states, labels.conj().T)) ** 2
    return np.max(fidelities, axis=2), np.argmax(fidelities, axis=2)


# --------------------------- Simulation settings -----------------------------
number_labels = 4
labels = state_labels(number_labels)

# Number of *intermediate* gates (virtual rotation layers)
number_gates = 2

# Gradient loop
number_iterations = 50
st = 0.01        # finite-difference step
lr = 0.006       # learning rate

# Synthetic dataset (psi, chi) clusters
np.random.seed(123)
number_clusters = 4
points_per_cluster = 10
N_points = number_clusters * points_per_cluster
centers = [
    (-0.29 * np.pi, 0.00 * np.pi),
    ( 0.00 * np.pi, 0.12 * np.pi),
    ( 0.29 * np.pi, 0.00 * np.pi),
    ( 0.00 * np.pi,-0.12 * np.pi),
]
width = 0.075
widths = [(width, width)] * 4

coordinates_cartesian = []
coordinates_jones = []

for i in range(number_clusters):
    for _ in range(points_per_cluster):
        psi, chi = np.random.normal(loc=centers[i], scale=widths[i])
        coordinates_cartesian.append([psi, chi, 0])
        coordinates_jones.append(Stokes2Jones(psi, chi))

coordinates_cartesian = np.array(coordinates_cartesian)
coordinates_jones = np.array(coordinates_jones, dtype=complex)


# ------------------------- Virtual waveplate models --------------------------
# Jones matrices in the H/V basis with fast-axis angle th (radians).
# These match the original script definitions.
quarter = lambda th: np.array(
    [
        [cos(th) ** 2 + 1j * sin(th) ** 2, (1 - 1j) * sin(th) * cos(th)],
        [(1 - 1j) * sin(th) * cos(th), sin(th) ** 2 + 1j * cos(th) ** 2],
    ],
    dtype=complex,
)

half = lambda th: np.array(
    [
        [cos(th) ** 2 - sin(th) ** 2, 2 * sin(th) * cos(th)],
        [2 * sin(th) * cos(th), sin(th) ** 2 - cos(th) ** 2],
    ],
    dtype=complex,
)


# ------------------------------ Background map -------------------------------
# Optional: plot trajectories on top of a precomputed cost map.
# If the image is missing, we’ll skip the background without failing.
fig, ax = plt.subplots()
bg_path = "cost_4_labels.png"
if os.path.exists(bg_path):
    bg_img = mpimg.imread(bg_path)
    ax.imshow(bg_img, extent=[0, 2 * np.pi, 0, 2 * np.pi], aspect="auto")


# ------------------------------ Optimization ---------------------------------
N_rec = 10                     # number of random restarts
final_cost_list = []
final_angles = []

colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

for i_rec in range(N_rec):
    print("restart #", i_rec)

    # Random initialization of gate angles
    params = np.random.uniform(size=(number_gates)) * 2 * np.pi

    # Initial stack (here: all quarter-wave for simplicity, as in the original code)
    stack = np.eye(2, dtype=complex)
    for k, th in enumerate(params):
        # If you want QWP/HWP alternating, replace the second line by: half(th)
        stack = quarter(th) @ stack

    # Propagate all inputs through the stack
    quantum_states = [stack @ v for v in coordinates_jones]
    quantum_states = np.array(quantum_states, dtype=complex).reshape(N_points, 1, 2)

    # Initial cost and containers
    costf = cost_function(quantum_states, coordinates_cartesian, labels)
    cost_list = []
    _, max_arg = test(quantum_states, labels)

    # Finite-difference gradient descent
    angles_trace = []     # keep params evolution to plot trajectory
    Gd = np.zeros_like(params)

    for it in range(number_iterations):
        x = params.copy()
        angles_trace.append(x.copy())

        # Compute gradient by perturbation of each parameter
        for j in range(len(x)):
            x[j] += st

            # Rebuild stack with perturbed angles (still QWPs as in original)
            stack_x = np.eye(2, dtype=complex)
            for th in x:
                stack_x = quarter(th) @ stack_x

            # Propagate and compute cost
            qs_x = [stack_x @ v for v in coordinates_jones]
            qs_x = np.array(qs_x, dtype=complex).reshape(N_points, 1, 2)

            cost_x = cost_function(qs_x, coordinates_cartesian, labels)
            Gd[j] = (costf - cost_x) / st

        # Gradient step
        params += lr * Gd

        # Rebuild stack with updated params and recompute cost
        stack = np.eye(2, dtype=complex)
        for th in params:
            stack = quarter(th) @ stack

        quantum_states = [stack @ v for v in coordinates_jones]
        quantum_states = np.array(quantum_states, dtype=complex).reshape(N_points, 1, 2)

        _, max_arg = test(quantum_states, labels)
        costf = cost_function(quantum_states, coordinates_cartesian, labels)
        cost_list.append(costf)

    final_cost_list.append(costf)
    final_angles.append(params)

    # Plot the trajectory on top of the background map
    angles_trace = np.array(angles_trace) % (2 * np.pi)
    if angles_trace.shape[1] == 2:
        x_coords, y_coords = angles_trace[:, 0], angles_trace[:, 1]
        ax.plot(x_coords, y_coords, marker='.', linestyle=' ', color=colors[i_rec % len(colors)])


# ------------------------------ Final plot -----------------------------------
final_angles_cost = np.hstack((final_angles, np.array(final_cost_list).reshape(-1, 1)))

ax.set_xlabel('Half angle')
ax.set_ylabel('Quarter angle')
ax.set_title('Cost function (trajectories)')
ax.grid(True)
plt.show()

# If you want to persist arrays, uncomment as needed:
# np.save(os.path.join(FOLDER, "coordinates_cartesian.npy"), coordinates_cartesian)
# np.save(os.path.join(FOLDER, "final_angles_cost.npy"), final_angles_cost)
