# -*- coding: utf-8 -*-
"""
Run the experiment for the photonic polarization classifier:
- Controls Zaber (quarter/half) and Standa (layered) rotators
- Reads polarization from Thorlabs PAX1000
- Builds dataset, runs gradient steps, and logs results
"""

import os
import time
import ctypes
import numpy as np
import matplotlib.pyplot as plt

from ctypes import byref, c_int, c_double, c_ulong, c_char_p, POINTER, cast
from scipy.spatial import cKDTree
from zaber.serial import AsciiSerial, AsciiDevice

# Standa/ximc low-level API (provides `lib`, types, and helpers)
from pyximc import *  # noqa: F401,F403  (kept as-is; module exists in this repo)

# Project modules with new names
from cost import cost_function
from labels_sphere import state_labels

# Convenient aliases (used in Stokes2Jones)
from numpy import cos as cos
from numpy import sin as sin
from numpy import pi as pi


# --------------------------- Output folder -----------------------------------
fol_name = input("Folder name: ").strip() or "run"
os.makedirs(fol_name, exist_ok=True)


# --------------------- Helpers: layer angle <-> position ---------------------
def deg_to_pos_middle(degrees):
    # NOTE: 'degrees' here are actually radians (kept from original script)
    conversion_factor = 14400 / (np.pi)
    position = conversion_factor * degrees
    return int(position)


def pos_to_deg_middle(position):
    conversion_factor = 1 / (14400 / (np.pi))
    degrees = conversion_factor * position
    return degrees


# ------------------------------ Classifier utils -----------------------------
def test(quantum_states, labels):
    """
    Given quantum_states (N_points x 1x2 Jones vectors) and labels (N_labels x 1x2),
    compute |<state|label>|^2 and return max fidelities and argmax indices.
    """
    fidelities = np.abs(np.matmul(quantum_states, labels.conj().T)) ** 2
    max_fidelities = np.max(fidelities, axis=2)
    arg_fidelities = np.argmax(fidelities, axis=2)
    return max_fidelities, arg_fidelities


# ---------------------------- Polarimetry helpers ----------------------------
def measure_quantum_state():
    """Average N readings from the PAX1000 → (psi, chi) [radians]."""
    N = 5
    psi = 0.0
    chi = 0.0
    for _ in range(N):
        scanID = c_int()
        lib_pax.TLPAX_getLatestScan(instrumentHandle, byref(scanID))

        azimuth = c_double()
        ellipticity = c_double()
        lib_pax.TLPAX_getPolarization(
            instrumentHandle, scanID.value, byref(azimuth), byref(ellipticity)
        )

        lib_pax.TLPAX_releaseScan(instrumentHandle, scanID)
        time.sleep(0.1)

        psi += azimuth.value
        chi += ellipticity.value

    return psi / N, chi / N


def Stokes2Jones(psi, chi):
    """
    Convert measured Stokes angles (psi, chi) into Jones coefficients.
    Returns a length-2 complex list: [c1, c2].
    """
    stokes_vector = [1, cos(2 * psi) * cos(2 * chi), sin(2 * psi) * cos(2 * chi), sin(2 * chi)]

    Q = stokes_vector[1]
    U = stokes_vector[2]
    V = stokes_vector[3]

    x = 2 * Q
    y = 2 * U
    z = 2 * V

    norm = np.sqrt(x**2 + y**2 + z**2)
    x_norm, y_norm, z_norm = x / norm, y / norm, z / norm

    phi = np.arctan2(y_norm, x_norm)
    theta = np.arccos(z_norm)

    quantum_state = [cos(theta / 2), np.exp(complex(0, 1) * phi) * sin(theta / 2)]
    return quantum_state


# ------------------------------ PAX1000 setup --------------------------------
# Adjust path if needed for your system
lib_pax = ctypes.cdll.LoadLibrary(
    r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLPAX_64.dll"
)

instrumentHandle = c_ulong()
IDQuery = True
resetDevice = False
resource = c_char_p(b"")
deviceCount = c_int()

lib_pax.TLPAX_findRsrc(instrumentHandle, byref(deviceCount))
if deviceCount.value < 1:
    print("No PAX1000 device found.")
else:
    print(deviceCount.value, "PAX1000 device(s) found.\n")

lib_pax.TLPAX_getRsrcName(instrumentHandle, 0, resource)
if 0 == lib_pax.TLPAX_init(resource.value, IDQuery, resetDevice, byref(instrumentHandle)):
    print("Connection to first PAX1000 initialized.")
else:
    print("Error with initialization.")
print("")

time.sleep(2)

# Settings
lib_pax.TLPAX_setMeasurementMode(instrumentHandle, 9)
lib_pax.TLPAX_setWavelength(instrumentHandle, c_double(808e-9))
lib_pax.TLPAX_setBasicScanRate(instrumentHandle, c_double(60))

# Echo settings
wavelength = c_double()
lib_pax.TLPAX_getWavelength(instrumentHandle, byref(wavelength))
print("Set wavelength [nm]: ", wavelength.value * 1e9)
mode = c_int()
lib_pax.TLPAX_getMeasurementMode(instrumentHandle, byref(mode))
print("Set mode: ", mode.value)
scanrate = c_double()
lib_pax.TLPAX_getBasicScanRate(instrumentHandle, byref(scanrate))
print("Set scanrate: ", scanrate.value, "\n")

time.sleep(5)


# -------------------------- Zaber (quarter/half) -----------------------------
rotadores = AsciiSerial("COM14", timeout=10, inter_char_timeout=0.01)
quarter_a = AsciiDevice(rotadores, 1)
half_a = AsciiDevice(rotadores, 2)
# quarter_a.home(); half_a.home()
# quarter_a.poll_until_idle(); half_a.poll_until_idle()


# -------------------------- Calibration lookup table -------------------------
# File moved to calibration/ with new name
calib_path = os.path.join("calibration", "cal_def_200x200_february_2024.txt")
# Keep same loading signature as original script (assumes file matches).
# If your file is columns -> add `unpack=True`.
s_alpha, s_beta, s_psi, s_chi = np.loadtxt(calib_path)
tree = cKDTree(np.c_[s_psi.ravel(), s_chi.ravel()])


# ------------------------------ Standa (ximc) --------------------------------
cur_dir = os.path.abspath(os.path.dirname(__file__))
ximc_dir = os.path.join(cur_dir, "ximc")

print("Library loaded")

sbuf = create_string_buffer(64)
lib.ximc_version(sbuf)
print("Library version: " + sbuf.raw.decode().rstrip("\0"))

# Bindy/network key (optional if no network devices)
result = lib.set_bindy_key(os.path.join(ximc_dir, "win32", "keyfile.sqlite").encode("utf-8"))
if result != Result.Ok:
    lib.set_bindy_key("keyfile.sqlite".encode("utf-8"))

# Order of plates/layers (kept from original)
plates_order = [
    "b'Axis 6-1'", "b'Axis 6-2'", "b'Axis 5-1'", "b'Axis 5-2'",
    "b'Axis 4-1'", "b'Axis 4-2'", "b'Axis 3-1'", "b'Axis 3-2'",
    "b'Axis 2-1'", "b'Axis 2-2'", "b'Axis 1-1'", "b'Axis 1-2'"
]

probe_flags = EnumerateFlags.ENUMERATE_PROBE + EnumerateFlags.ENUMERATE_NETWORK
device_id, friendly_name, layers = [], [], []

# First block
enum_hints = b"addr=11.0.0.2"
print(probe_flags, enum_hints)
devenum = lib.enumerate_devices(probe_flags, enum_hints)
print("Device enum handle:", repr(devenum))
print("Device enum handle type:", repr(type(devenum)))

dev_count = lib.get_device_count(devenum)
print("Device count:", repr(dev_count))
controller_name = controller_name_t()
for dev_ind in range(0, dev_count):
    enum_name = lib.get_device_name(devenum, dev_ind)
    result = lib.get_enumerate_device_controller_name(devenum, dev_ind, byref(controller_name))
    open_name = lib.get_device_name(devenum, dev_ind)
    device_id.append(lib.open_device(open_name))

    if result == Result.Ok:
        print(
            f"Enumerated device #{dev_ind} name (port name): {repr(enum_name)}. "
            f"Friendly name: {repr(controller_name.ControllerName)}."
        )
        friendly_name.append(str(controller_name.ControllerName))
    else:
        print("I can't find the device")

for i in plates_order:
    idx = friendly_name.index(i)
    layers.append(idx)

# Second block
probe_flags = EnumerateFlags.ENUMERATE_PROBE + EnumerateFlags.ENUMERATE_NETWORK
friendly_name = []
enum_hints = b"addr=10.0.0.2"

print(probe_flags, enum_hints)
devenum = lib.enumerate_devices(probe_flags, enum_hints)
print("Device enum handle:", repr(devenum))
print("Device enum handle type:", repr(type(devenum)))

dev_count = lib.get_device_count(devenum)
print("Device count:", repr(dev_count))
controller_name = controller_name_t()
for dev_ind in range(0, dev_count):
    enum_name = lib.get_device_name(devenum, dev_ind)
    result = lib.get_enumerate_device_controller_name(devenum, dev_ind, byref(controller_name))
    open_name = lib.get_device_name(devenum, dev_ind)
    device_id.append(lib.open_device(open_name))
    if result == Result.Ok:
        print(
            f"Enumerated device #{dev_ind} name (port name): {repr(enum_name)}. "
            f"Friendly name: {repr(controller_name.ControllerName)}."
        )
        friendly_name.append(str(controller_name.ControllerName))
    else:
        print("I can't find the device")

for i in plates_order:
    idx = friendly_name.index(i)
    layers.append(idx + 12)

# Motion settings
mvst = move_settings_t()
mvst.Speed = 5000
mvst.Accel = 5000
mvst.Decel = 5000


# -------------------------- Algorithm initialization -------------------------
print("Initialization of algorithm")

initial_values = [
    0, 0, 26519, 5989, 21363, 7126, 6599, 21481, 27765, 4642, 20687, 14863, 13872,
    10630, 22906, 4981, 28391, 3998, 23089, 7565, 28715, 21530, 7717, 23283, 12525, 21494
]

print("Setting fast axes to zero...")
for i in range(len(layers)):
    device = device_id[layers[i]]
    lib.set_move_settings(device, byref(mvst))
    lib.command_move(device, initial_values[i + 2], 0)
    lib.command_wait_for_stop(device, 10)
for _ in range(len(layers)):
    lib.command_wait_for_stop(device, 10)
time.sleep(1)


# ------------------------------- Experiment data -----------------------------
number_labels = 4
labels = state_labels(number_labels)

number_gates = 2       # intermediate rotation layers (not counting initialization)
number_iterations = 30
st = 0.01              # finite-difference step
lr = 0.0055            # learning rate

# Synthetic points in (psi, chi)
np.random.seed(123)
number_clusters = 4
points_per_cluster = 10
N_points = number_clusters * points_per_cluster
centers = [(-0.29 * np.pi, 0 * np.pi), (0 * np.pi, 0.12 * np.pi),
           (0.29 * np.pi, 0 * np.pi), (0 * np.pi, -0.12 * np.pi)]
width = 0.075
widths = [(width, width)] * 4

coordinates_cartesian, coordinates_layers, coordinates_measured, coordinates_jones = [], [], [], []

for i in range(number_clusters):
    for _ in range(points_per_cluster):
        point = np.random.normal(loc=centers[i], scale=widths[i])
        coordinates_cartesian.append([point[0], point[1], 0])
        coordinates_jones.append(Stokes2Jones(point[0], point[1]))
        # Lookup nearest calibration entry
        dd, ii = tree.query([point[0], point[1]], k=1)
        point_layers = [int(s_alpha[ii]), int(s_beta[ii]), 0]
        coordinates_layers.append(point_layers)

coordinates_cartesian = np.array(coordinates_cartesian)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.scatter(coordinates_cartesian[:, 0] / np.pi, coordinates_cartesian[:, 1] / np.pi, c="r", s=20)
plt.xlim([-0.5, 0.5])
plt.ylim([-0.25, 0.25])
plt.xlabel(r'$\psi$ [units of $\pi$]')
plt.ylabel(r'$\chi$ [units of $\pi$]')
plt.rcParams["mathtext.fontset"] = "cm"
plt.show()


# ------------------------------ Sanity check ---------------------------------
coordinates_layers = np.array(coordinates_layers, dtype=int)
print("coordinates cartesian = \n", coordinates_cartesian)
print("coordinates layers = \n", coordinates_layers)

quantum_states_list, coordinates_measured_list = [], []
cost_list, max_arg_list, inter_positions_list = [], [], []

print("Checking the initial quantum states ...")
quantum_states, coordinates_measured, inter_positions = [], [], []

for i in range(len(coordinates_layers)):
    quarter_a.move_abs(coordinates_layers[i][0])
    half_a.move_abs(coordinates_layers[i][1])
    quarter_a.poll_until_idle(); half_a.poll_until_idle()
    time.sleep(0.1)

    psi_i, chi_i = measure_quantum_state()
    quantum_state_i = Stokes2Jones(psi_i, chi_i)

    coordinates_measured.append([psi_i, chi_i])
    quantum_states.append(quantum_state_i)

print(coordinates_measured)

coordinates_measured_plot = np.array(coordinates_measured)
coordinates_cartesian = np.array(coordinates_cartesian)
number_points = len(coordinates_cartesian)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.scatter(coordinates_cartesian[:, 0] / np.pi, coordinates_cartesian[:, 1] / np.pi, c="r", s=20)
ax.scatter(coordinates_measured_plot[:, 0] / np.pi, coordinates_measured_plot[:, 1] / np.pi, c="g", s=20)
plt.xlim([-0.5, 0.5])
plt.ylim([-0.25, 0.25])
plt.xlabel(r'$\psi$ [units of $\pi$]')
plt.ylabel(r'$\chi$ [units of $\pi$]')
plt.rcParams["mathtext.fontset"] = "cm"
plt.legend(['Theoretical', 'Experimental'])
plt.show()


# -------------------------------- Algorithm ----------------------------------
N_rec = 10
final_cost_list, final_angles, initial_angles = [], [], []

np.random.seed(222)

for i_rec in range(N_rec):
    print("recorrido No.", i_rec)

    params = np.random.uniform(size=(number_gates)) * 2 * np.pi  # random init angles
    initial_angles.append(params)

    cost_value = cost_function(quantum_states, coordinates_cartesian, labels)
    costf = cost_value

    max_fidel, max_arg = test(quantum_states, labels)

    quantum_states_list.append(quantum_states)
    coordinates_measured_list.append(coordinates_measured)
    max_arg_list.append(max_arg)

    # Read intermediate layer positions
    for i in range(len(layers)):
        device = device_id[layers[i]]
        x_pos = get_position_t()
        lib.get_position(device, byref(x_pos))
        inter_positions.append(x_pos.Position)
    inter_positions_list.append(inter_positions)

    # Prepare for first step
    quantum_states, coordinates_measured, inter_positions = [], [], []
    Gd = np.zeros(len(params))

    print("First step of the algorithm ...")
    for i in range(number_gates):
        device = device_id[layers[i]]
        lib.set_move_settings(device, byref(mvst))
        position = deg_to_pos_middle(params[i])
        lib.command_move(device, position, 0)
        lib.command_wait_for_stop(device, 10)
    for _ in range(number_gates):
        lib.command_wait_for_stop(device, 10)

    for i in range(number_points):
        quarter_a.move_abs(coordinates_layers[i][0])
        half_a.move_abs(coordinates_layers[i][1])
        quarter_a.poll_until_idle(); half_a.poll_until_idle()
        time.sleep(0.1)

        psi_i, chi_i = measure_quantum_state()
        quantum_state_i = Stokes2Jones(psi_i, chi_i)

        quantum_states.append(quantum_state_i)
        coordinates_measured.append([psi_i, chi_i])
        time.sleep(0.1)

    cost_value = cost_function(quantum_states, coordinates_cartesian, labels)
    costf = cost_value
    cost_list.append(costf)

    max_fidel, max_arg = test(quantum_states, labels)
    quantum_states_list.append(quantum_states)
    coordinates_measured_list.append(coordinates_measured)
    max_arg_list.append(max_arg)

    for i in range(len(layers)):
        device = device_id[layers[i]]
        x_pos = get_position_t()
        lib.get_position(device, byref(x_pos))
        inter_positions.append(x_pos.Position)
    inter_positions_list.append(inter_positions)

    print("End of first step of the algorithm.")

    # Epochs
    for index in range(number_iterations):
        print("*************************************************")
        x = params.copy()

        # Finite-difference gradient
        for j in range(len(x)):
            print(index, "mini epoch layer number:", j)
            x[j] += st

            quantum_states, coordinates_measured, inter_positions = [], [], []

            # Move all intermediate layers
            for i in range(number_gates):
                device = device_id[layers[i]]
                lib.set_move_settings(device, byref(mvst))
                position = deg_to_pos_middle(x[i])
                lib.command_move(device, position, 0)
                lib.command_wait_for_stop(device, 10)
            for _ in range(number_gates):
                lib.command_wait_for_stop(device, 10)

            # Measure all points
            for i in range(number_points):
                quarter_a.move_abs(coordinates_layers[i][0])
                half_a.move_abs(coordinates_layers[i][1])
                quarter_a.poll_until_idle(); half_a.poll_until_idle()
                time.sleep(0.1)

                psi_i, chi_i = measure_quantum_state()
                quantum_state_i = Stokes2Jones(psi_i, chi_i)

                quantum_states.append(quantum_state_i)
                coordinates_measured.append([psi_i, chi_i])
                time.sleep(0.1)

            cost_value = cost_function(quantum_states, coordinates_cartesian, labels)
            print("cost_value =", cost_value)
            Gd[j] = (costf - cost_value) / st

        # Update params
        params += lr * Gd

        # Apply new params
        for i in range(number_gates):
            device = device_id[layers[i]]
            lib.set_move_settings(device, byref(mvst))
            position = deg_to_pos_middle(params[i])
            lib.command_move(device, position, 0)
            lib.command_wait_for_stop(device, 10)
        for _ in range(number_gates):
            lib.command_wait_for_stop(device, 10)

        # Measure again with updated params
        quantum_states, coordinates_measured, inter_positions = [], [], []
        for i in range(number_points):
            quarter_a.move_abs(coordinates_layers[i][0])
            half_a.move_abs(coordinates_layers[i][1])
            quarter_a.poll_until_idle(); half_a.poll_until_idle()
            time.sleep(0.1)

            psi_i, chi_i = measure_quantum_state()
            quantum_state_i = Stokes2Jones(psi_i, chi_i)

            quantum_states.append(quantum_state_i)
            coordinates_measured.append([psi_i, chi_i])
            time.sleep(0.1)

        max_fidel, max_arg = test(quantum_states, labels)
        print("max_fidel = ", max_fidel)
        print("max_arg   = ", max_arg)

        costf = cost_function(quantum_states, coordinates_cartesian, labels)
        print("costf = ", costf)

        cost_list.append(costf)
        quantum_states_list.append(quantum_states)
        coordinates_measured_list.append(coordinates_measured)
        max_arg_list.append(max_arg)

        for i in range(len(layers)):
            device = device_id[layers[i]]
            x_pos = get_position_t()
            lib.get_position(device, byref(x_pos))
            inter_positions.append(x_pos.Position)
        inter_positions_list.append(inter_positions)

    final_cost_list.append(costf)
    final_angles.append(params)

initial_final_angles_cost = np.hstack(
    (initial_angles, final_angles, np.array(final_cost_list).reshape(-1, 1))
)

# Save minimal summary (others commented remain available)
ruta_archivo = os.path.join(fol_name, "initial_final_angles_cost.npy")
np.save(ruta_archivo, np.array(initial_final_angles_cost))


# --------------------------------- Close -------------------------------------
lib_pax.TLPAX_reset(instrumentHandle)
lib_pax.TLPAX_close(instrumentHandle)
print("Connection to PAX1000 closed.")

rotadores.close()
print("Connection to Zaber rotators closed.")

for i in range(len(layers)):
    device = device_id[layers[i]]
    lib.close_device(byref(cast(device, POINTER(c_int))))
print("Connection to Standa rotators closed.")
