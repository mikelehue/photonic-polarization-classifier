# Function to transform cartesian cooridnates to quantum coefficients and get quantum state for each data point
import numpy as np

def cart_to_coef_function(points_x, points_y, points_z):
    phi_once = np.arctan2(points_y, points_x)

    theta_once = np.arccos(points_z)

    c1_once = np.cos(theta_once / 2)

    c2_once = np.exp(complex(0, 1) * phi_once) * np.sin(theta_once / 2)

    return np.array(c1_once, dtype=complex), np.array(c2_once, dtype=complex)
