# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:00:56 2023

@author: QNanoLab1
"""
import numpy as np
from zaber.serial import AsciiSerial, AsciiDevice
from scipy.spatial import cKDTree
import ctypes
from ctypes import *
import matplotlib.pyplot as plt
import time
import random as rn
from pyximc import *
import os
from cost import cost_function
from Labels_sphere import state_labels
from numpy import cos as cos
from numpy import sin as sin
from numpy import pi as pi

# Folder for saving the data
fol_name = input("Folder name: ")
if not os.path.exists(fol_name):
    os.mkdir(fol_name)

###################### Steps in the middle of the algorithm ###################

def deg_to_pos_middle(degrees):
    conversion_factor = 14400 / (np.pi)
    position = conversion_factor * degrees
    return position

def pos_to_deg_middle(position):
    conversion_factor = 1 / (14400 / (np.pi))
    degrees = conversion_factor * position
    return degrees

# def test(quantum_states, labels):
#     fidelities = np.matmul(quantum_states,labels)
#     fidelities = fidelities*np.conj(fidelities)
#     first_label = fidelities[0]
#     second_label = fidelities[1]
#     max_fidelities = [np.max([first_label[i], second_label[i]]) for i in range(len(fidelities[0]))]
#     arg_fidelities = [np.argmax([first_label[i], second_label[i]]) for i in range(len(fidelities[0]))]

#     return max_fidelities, arg_fidelities

def test(quantum_states, labels):
    fidelities = np.abs(np.matmul(quantum_states, labels.conj().T))**2
    num_labels = labels.shape[0]
    max_fidelities = np.max(fidelities, axis=2)
    arg_fidelities = np.argmax(fidelities, axis=2)

    return max_fidelities, arg_fidelities

###################### Inicialization of devices ##############################

def measure_quantum_state():

    N = 5

    psi = 0
    chi = 0
    
    for x in range(N):    

        revolutionCounter = c_int()
        scanID = c_int()
        lib_pax.TLPAX_getLatestScan(instrumentHandle, byref(scanID))
    
        azimuth = c_double()
        ellipticity = c_double()
        lib_pax.TLPAX_getPolarization(instrumentHandle, scanID.value, byref(azimuth), byref(ellipticity))
        # print("Azimuth [rad]: ", azimuth.value)
        # print("Ellipticity [rad]: ", ellipticity.value)
        # print("")
    
        lib_pax.TLPAX_releaseScan(instrumentHandle, scanID)
        time.sleep(0.1)
    
        psi += azimuth.value
        chi += ellipticity.value
        
    psi /= N
    chi /= N

    return psi, chi

def Stokes2Jones(psi, chi):

    stokes_vector = [1, cos(2*psi)*cos(2*chi), sin(2*psi)*cos(2*chi),  sin(2*chi)]
    
    # print(stokes_vector)
    
    Q=stokes_vector[1];
    U=stokes_vector[2];
    V=stokes_vector[3];
    

    x = 2*Q
    y = 2*U
    z = 2*V
    
    norm = np.sqrt(x**2 + y**2 + z**2)
    
    x_norm = x/norm
    y_norm = y/norm
    z_norm = z/norm
    
    phi = np.arctan2(y_norm, x_norm)
    theta = np.arccos(z_norm)
    
    quantum_state = [cos(theta/2), np.exp(complex(0,1)*phi)*sin(theta/2)]
    
    return quantum_state

# Load DLL library
lib_pax = cdll.LoadLibrary("C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLPAX_64.dll")

# Detect and initialize PAX1000 device
instrumentHandle = c_ulong()
IDQuery = True
resetDevice = False
resource = c_char_p(b"")
deviceCount = c_int()

# Check how many PAX1000 are connected
lib_pax.TLPAX_findRsrc(instrumentHandle, byref(deviceCount))
if deviceCount.value < 1 :
    print("No PAX1000 device found.")
    # exit()
else:
    print(deviceCount.value, "PAX1000 device(s) found.")
    print("")

# Connect to the first available PAX1000
lib_pax.TLPAX_getRsrcName(instrumentHandle, 0, resource)
if (0 == lib_pax.TLPAX_init(resource.value, IDQuery, resetDevice, byref(instrumentHandle))):
    print("Connection to first PAX1000 initialized.")
else:
    print("Error with initialization.")
    # exit()
print("")

# Short break to make sure the device is correctly initialized
time.sleep(2)

# Make settings
lib_pax.TLPAX_setMeasurementMode(instrumentHandle, 9)
lib_pax.TLPAX_setWavelength(instrumentHandle, c_double(808e-9))
lib_pax.TLPAX_setBasicScanRate(instrumentHandle, c_double(60))

# Check settings
wavelength = c_double()
lib_pax.TLPAX_getWavelength(instrumentHandle, byref(wavelength))
print("Set wavelength [nm]: ", wavelength.value*1e9)
mode = c_int()
lib_pax.TLPAX_getMeasurementMode(instrumentHandle, byref(mode))
print("Set mode: ", mode.value)
scanrate = c_double()
lib_pax.TLPAX_getBasicScanRate(instrumentHandle, byref(scanrate))
print("Set scanrate: ", scanrate.value)
print("")

# Short break
time.sleep(5)

rotadores = AsciiSerial("COM14", timeout=10, inter_char_timeout=0.01)
quarter_a = AsciiDevice(rotadores, 1)
half_a = AsciiDevice(rotadores, 2)

# quarter_a.home()
# half_a.home()

# quarter_a.poll_until_idle()
# half_a.poll_until_idle()

s_alpha, s_beta, s_psi, s_chi = np.loadtxt('cal_def_200x200_january_2024.txt')
tree = cKDTree(np.c_[s_psi.ravel(), s_chi.ravel()])

cur_dir = os.path.abspath(os.path.dirname(__file__))  # Specifies the current directory.
ximc_dir = os.path.join(cur_dir, "ximc")  # Formation of the directory name with all dependencies.

# note that ximc uses stdcall on win
print("Library loaded")

sbuf = create_string_buffer(64)
lib.ximc_version(sbuf)
print("Library version: " + sbuf.raw.decode().rstrip("\0"))


# Set bindy (network) keyfile. Must be called before any call to "enumerate_devices" or "open_device" if you
# wish to use network-attached controllers. Accepts both absolute and relative paths, relative paths are resolved
# relative to the process working directory. If you do not need network devices then "set_bindy_key" is optional.
# In Python make sure to pass byte-array object to this function (b"string literal").
result = lib.set_bindy_key(os.path.join(ximc_dir, "win32", "keyfile.sqlite").encode("utf-8"))
if result != Result.Ok:
    lib.set_bindy_key("keyfile.sqlite".encode("utf-8"))  # Search for the key file in the current directory.
 
# define the plates order    
plates_order = ["b'Axis 6-1'", "b'Axis 6-2'", "b'Axis 5-1'", "b'Axis 5-2'", "b'Axis 4-1'", "b'Axis 4-2'", "b'Axis 3-1'", "b'Axis 3-2'", "b'Axis 2-1'", "b'Axis 2-2'", "b'Axis 1-1'", "b'Axis 1-2'"]

probe_flags = EnumerateFlags.ENUMERATE_PROBE + EnumerateFlags.ENUMERATE_NETWORK

device_id = []
friendly_name = []
layers = []

enum_hints = b"addr=11.0.0.2"

print(probe_flags, enum_hints)
devenum = lib.enumerate_devices(probe_flags, enum_hints)
print("Device enum handle: " + repr(devenum))
print("Device enum handle type: " + repr(type(devenum)))

dev_count = lib.get_device_count(devenum)
print("Device count: " + repr(dev_count))
controller_name = controller_name_t()
for dev_ind in range(0, dev_count):
    enum_name = lib.get_device_name(devenum, dev_ind)
    result = lib.get_enumerate_device_controller_name(devenum, dev_ind, byref(controller_name))
    open_name = lib.get_device_name(devenum, dev_ind)
    device_id.append(lib.open_device(open_name))

    if result == Result.Ok:
        print("Enumerated device #{} name (port name): ".format(dev_ind) + repr(
            enum_name) + ". Friendly name: " + repr(
            controller_name.ControllerName) + ".")
        friendly_name.append(str(controller_name.ControllerName))
        # print('device_id = ', device_id)
        # print('friendly_name = ', friendly_name)
    else:
        print('I cant find the device')

for i in plates_order:
    # print('plates_order = ', i)
    idx = friendly_name.index(i)
    layers.append(idx)
    # print('layers =', layers)

# This is device search and enumeration with probing. It gives more information about devices.
probe_flags = EnumerateFlags.ENUMERATE_PROBE + EnumerateFlags.ENUMERATE_NETWORK

enum_hints = b"addr=10.0.0.2"

# device_id = []
friendly_name = []

# enum_hints = b"addr=" # Use this hint string for broadcast enumerate
print(probe_flags, enum_hints)
devenum = lib.enumerate_devices(probe_flags, enum_hints)
print("Device enum handle: " + repr(devenum))
print("Device enum handle type: " + repr(type(devenum)))

dev_count = lib.get_device_count(devenum)
print("Device count: " + repr(dev_count))
controller_name = controller_name_t()
for dev_ind in range(0, dev_count):
    enum_name = lib.get_device_name(devenum, dev_ind)
    result = lib.get_enumerate_device_controller_name(devenum, dev_ind, byref(controller_name))
    open_name = lib.get_device_name(devenum, dev_ind)
    device_id.append(lib.open_device(open_name))
    if result == Result.Ok:
        print("Enumerated device #{} name (port name): ".format(dev_ind) + repr(
            enum_name) + ". Friendly name: " + repr(
            controller_name.ControllerName) + ".")
        friendly_name.append(str(controller_name.ControllerName))
        # print('device_id = ', device_id)
        # print('friendly_name = ', friendly_name)
    else:
        print('I cant find the device')

for i in plates_order:
#    print('plates_order = ', i)
    idx = friendly_name.index(i)
    layers.append(idx+12)
    
# Filling in the structure move_settings_t    
mvst = move_settings_t()
mvst.Speed = 5000
mvst.Accel = 5000
mvst.Decel = 5000

################### Inicialization on algorithm ###############################

print('Initialization of algorithm')

initial_values = [0,0,26519, 5989, 21363, 7126, 6599, 21481, 27765, 4642, 20687, 14863, 13872,
                  10630, 22906, 4981, 28391, 3998, 23089, 7565, 28715, 21530, 7717, 23283, 12525, 21494]

print('Setting fasts axis to zero...')

for i in range(len(layers)):
    device = device_id[layers[i]]
    lib.set_move_settings(device, byref(mvst))
    lib.command_move(device, initial_values[i+2], 0)
    lib.command_wait_for_stop(device, 10)

for i in range(len(layers)):
    lib.command_wait_for_stop(device, 10)

time.sleep(1)

############# DATA FOR THE ALGORITHM ##########################################

#Choose number of labels
number_labels = 4
labels = state_labels(number_labels)
 
#Choose number of gates
number_gates = 2  #Initialization gates do not count

#Choose the number of times the whole set of gates is applied
number_iterations = 30

#Choose the step for calculate the gradient
st = 0.01

#Choose the value of the learning rate
lr = 0.0055

###############################################################################
#Initialize data set
np.random.seed(123)
# Set up cluster parameters
number_clusters = 4
points_per_cluster = 10
N_points = number_clusters * points_per_cluster
centers = [(-0.29*np.pi, 0*np.pi), (0*np.pi, 0.12*np.pi), (0.29*np.pi, 0*np.pi), (0*np.pi, -0.12*np.pi)]
width = 0.075
widths = [(width, width), (width, width), (width, width), (width, width)]

# Initialize arrays for coordinates and layers
coordinates_cartesian = []
coordinates_layers = []
coordinates_measured = []
coordinates_jones = []

# Generate points within clusters and transform them to moves within layers
for i in range(number_clusters):
    # Generate points within current cluster
    for j in range(points_per_cluster):
        # Generate point with Gaussian distribution
        point = np.random.normal(loc=centers[i], scale=widths[i])
        coordinates_cartesian.append([point[0], point[1], 0])
        coordinates_jones.append(Stokes2Jones(point[0],point[1]))
        # Transform to moves within layers using lookup table
        dd, ii = tree.query([point[0], point[1]], k=1)
        point = [int(s_alpha[ii]), int(s_beta[ii]), 0]
        coordinates_layers.append(point)

# Convert coordinates to numpy arrays
coordinates_cartesian = np.array(coordinates_cartesian)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.scatter(coordinates_cartesian[:, 0]/np.pi, coordinates_cartesian[:, 1]/np.pi, c="r", s=20)
plt.xlim([-0.5, 0.5])
plt.ylim([-0.25, 0.25])
plt.xlabel(r'$\psi$ [units of $\pi$]')
plt.ylabel(r'$\chi$ [units of $\pi$]')
plt.rcParams["mathtext.fontset"] = "cm"
plt.show()

################### Check init stats ########## ###############################

coordinates_layers = np.array(coordinates_layers, dtype = int)
print('coordinates cartesian = ')
print(coordinates_cartesian)
print('coordinates layers = ')
print(coordinates_layers)

quantum_states_list = []
coordinates_measured_list = []
cost_list = []
max_arg_list = []
inter_positions_list = []

# Check if the quantum states are well defined
print('Checking the initial quantum states ...')
quantum_states = []
coordinates_measured = []
inter_positions = []

for i in range(len(coordinates_layers)):
    
    quarter_a.move_abs(coordinates_layers[i][0])
    half_a.move_abs(coordinates_layers[i][1])

    quarter_a.poll_until_idle()
    half_a.poll_until_idle()

    time.sleep(0.1)
    
    psi_ask = coordinates_cartesian[i][0]
    chi_ask = coordinates_cartesian[i][1]
    
    print('psi_ask = ', psi_ask)
    print('chi_ask = ', chi_ask)
    
    # Measure quantum state

    psi_i, chi_i = measure_quantum_state()
    
    quantum_state_i = Stokes2Jones(psi_i, chi_i)

    print('phi_i, chi_i = ', psi_i, chi_i)
    coordinates_measured.append([psi_i, chi_i])
    
    # print('quantum_state_i = ', quantum_state_i)
    quantum_states.append(quantum_state_i)

# print('coordinates measured ...')
print(coordinates_measured)

coordinates_measured_plot = np.array(coordinates_measured)
coordinates_cartesian = np.array(coordinates_cartesian)
number_points = len(coordinates_cartesian)

print('quantum_states = ', quantum_states)




fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.scatter(coordinates_cartesian[:, 0]/np.pi, coordinates_cartesian[:, 1]/np.pi, c="r", s=20)
ax.scatter(coordinates_measured_plot[:, 0]/np.pi, coordinates_measured_plot[:, 1]/np.pi, c="g", s=20)
plt.xlim([-0.5, 0.5])
plt.ylim([-0.25, 0.25])
plt.xlabel(r'$\psi$ [units of $\pi$]')
plt.ylabel(r'$\chi$ [units of $\pi$]')
plt.rcParams["mathtext.fontset"] = "cm"
plt.legend(['Theoretical', 'Experimental'])
plt.show()


################### Algorithm #################################################
N_rec = 10

final_cost_list = []
final_angles = []
initial_angles = []

np.random.seed(222)

for i_rec in range(N_rec):
    
    print('recorrido No. ', i_rec)

    
    #Randomly initialize the angles of the gates
    params = np.random.uniform(size=(number_gates))*2*np.pi
     
    initial_angles.append(params)
    
    # Save first data
    cost_value = cost_function(quantum_states,coordinates_cartesian,labels)
    costf = cost_value
    
    max_fidel, max_arg = test(quantum_states, labels)
    
    quantum_states_list.append(quantum_states)
    coordinates_measured_list.append(coordinates_measured)
    max_arg_list.append(max_arg)
    
    # Get intermediate layers position
    
    for i in range(len(layers)):
        device = device_id[layers[i]]
        x_pos = get_position_t()
        lib.get_position(device, byref(x_pos))
        inter_positions.append(x_pos.Position)
        
    inter_positions_list.append(inter_positions)
    
    #Initialize quantum states list to calculate initial cost
    quantum_states = []
    coordinates_measured = []
    inter_positions = []
    #
    #Initialize gradient descent
    Gd = np.zeros(len(params))
    #Initialize params in the intermediate layers
    
    print('First step of the algorithm ...')
    
    for i in range(number_gates):
         # print('moving layer ', layers[i])
         device = device_id[layers[i]]
         lib.set_move_settings(device, byref(mvst))
         position = int(deg_to_pos_middle(params[i]))
         # print('to position', position)
         lib.command_move(device, position, 0)
         # x_status = status_t()
         # print('status ', lib.get_status(device, byref(x_status)))
         lib.command_wait_for_stop(device, 10)
    
    for i in range(number_gates):
         lib.command_wait_for_stop(device, 10)
    
    for i in range(number_points):
    
        quarter_a.move_abs(coordinates_layers[i][0])
        half_a.move_abs(coordinates_layers[i][1])
    
        quarter_a.poll_until_idle()
        half_a.poll_until_idle()
    
        time.sleep(0.1)
    
        # Measure quantum state
    
        psi_i, chi_i = measure_quantum_state()
        
        quantum_state_i = Stokes2Jones(psi_i, chi_i)
    
        # print('quantum_state_i = ', quantum_state_i)
        quantum_states.append(quantum_state_i)
        coordinates_measured.append([psi_i, chi_i])
        time.sleep(0.1)
    
    # print('quantum_states = ', quantum_states)
    # print('coordinates_cartesian = ', coordinates_cartesian)
    # print('labels = ', labels)
    
    #Initialize the list of costs
    cost_value = cost_function(quantum_states,coordinates_cartesian,labels)
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
    
    # print('cost list = ', cost_list)
    
    print('End of first step of the algorithm.')
    
    #Start iterative procedure. As many epochs as number_iterations
    for index in range(number_iterations):
        
        print('*************************************************')
        
        #Introduce the parameters of the rotation layers
        params_initial = params.copy()
        x = params.copy()
    
        #Create artificial gradient descent
        for j in range(len(x)):
    
            print(index, 'mini epoch layer number:  ', j)
    
            x[j] += st
    
            quantum_states= []
            coordinates_measured = []
            inter_positions = []
            #we move all the itermediate layers before initializing the points
    
            for i in range(number_gates):
                # print('moving layer ', layers[i])
                device = device_id[layers[i]]
                lib.set_move_settings(device, byref(mvst))
                position = int(deg_to_pos_middle(x[i]))
                # print('to position ', position)
                lib.command_move(device, position, 0)
                # x_status = status_t()
                # print('status ', lib.get_status(device, byref(x_status)))
                lib.command_wait_for_stop(device, 10)
    
            for i in range(number_gates):
                lib.command_wait_for_stop(device, 10)
    
    
            for i in range(number_points):
                    quarter_a.move_abs(coordinates_layers[i][0])
                    half_a.move_abs(coordinates_layers[i][1])
    
                    quarter_a.poll_until_idle()
                    half_a.poll_until_idle()
    
                    time.sleep(0.1)
    
                    # Measure quantum state
    
                    psi_i, chi_i = measure_quantum_state()
                    
                    quantum_state_i = Stokes2Jones(psi_i, chi_i)
                    
                    # print('quantum_state_i = ', quantum_state_i)
                    quantum_states.append(quantum_state_i)
                    coordinates_measured.append([psi_i, chi_i])
    
                    time.sleep(0.1)
    
            cost_value = cost_function(quantum_states,coordinates_cartesian,labels)
    
            print('cost_value = ', cost_value)
            Gd[j] = (costf - cost_value)/st
    
            # x[j] -= st
    
            # if we take into account the change in the cost based on the one before the iterations for the individual entries, we have to change the parameters
            # individually based all the time on the reference parameters entering the loop, and we cannot update the parameters individually in each iteration. If so,
            # we should also take that into account to calculate the Gd and include in the formula the cost generated by the entry i when looking at GD of i+1, and not the initial one
    
        ch = lr * Gd  # if we consider all the changes at the same time maybe it is better if I make this number smaller
    
        # We could either update the value of the entry of x step by step takin into account how much it varies per single step, or we could instead
        # (as it is done here) modify step by step th whole array of values of x an then compute the gradient based on that. In th end we are
        # multiplying matrices s this is linear. Maybe considering the whole array we do not get interactions between arrays in the sense that moving one gate affects the rest)
        # and it might be better (not to lose track of hat changes improve at what moment and what changes make it worse) /more precise to perform
        # the updating step by step.
    
        params += ch
    
    
        for i in range(number_gates):
                # print('moving layer ', layers[i])
                device = device_id[layers[i]]
                lib.set_move_settings(device, byref(mvst))
                position = int(deg_to_pos_middle(params[i]))
                # print('to position ', position)
                lib.command_move(device, position, 0)
                # x_status = status_t()
                # print('status ', lib.get_status(device, byref(x_status)))
                lib.command_wait_for_stop(device, 10)
                
        for i in range(number_gates):
            lib.command_wait_for_stop(device, 10)
    
        quantum_states = []
        coordinates_measured = []
        inter_positions = []
    
        for i in range(number_points):
            quarter_a.move_abs(coordinates_layers[i][0])
            half_a.move_abs(coordinates_layers[i][1])
    
            quarter_a.poll_until_idle()
            half_a.poll_until_idle()
    
            time.sleep(0.1)
    
            # Measure quantum state
    
            psi_i, chi_i = measure_quantum_state()
            
            quantum_state_i = Stokes2Jones(psi_i, chi_i)
    
            # print('quantum_state_i = ', quantum_state_i)
            quantum_states.append(quantum_state_i)
            coordinates_measured.append([psi_i, chi_i])
    
            time.sleep(0.1)
    
        max_fidel, max_arg = test(quantum_states, labels)
        # print('quantum_states = ', quantum_states)
        # print('labels = ', labels)
        print('max_fidel = ', max_fidel)
        print('max_arg = ', max_arg)
    
        costf = cost_function(quantum_states,coordinates_cartesian,labels)
    
        print('costf = ', costf)
    
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
    
initial_final_angles_cost = np.hstack((initial_angles, final_angles, np.array(final_cost_list).reshape(-1, 1)))

    # plt.figure()
    # plt.plot(cost_list)
    # plt.show()
    
    # ruta_archivo = os.path.join(fol_name, "quantum_states.npy")
    # quantum_states_list = np.array(quantum_states_list)
    # np.save(ruta_archivo, quantum_states_list)
    
    # ruta_archivo = os.path.join(fol_name, "coordinates_measured.npy")
    # coordinates_measured_list = np.array(coordinates_measured_list)
    # np.save(ruta_archivo, coordinates_measured_list)
    
    # ruta_archivo = os.path.join(fol_name, "inter_positions.npy")
    # inter_positions_list = np.array(inter_positions_list)
    # np.save(ruta_archivo, inter_positions_list)
    
    # ruta_archivo = os.path.join(fol_name, "max_arg.npy")
    # max_arg_list = np.array(max_arg_list)
    # np.save(ruta_archivo, max_arg_list)
    
    # ruta_archivo = os.path.join(fol_name, "coordinates_theoretical.npy")
    # coordinates_cartesian = np.array(coordinates_cartesian)
    # np.save(ruta_archivo, coordinates_cartesian)
    
    # ruta_archivo = os.path.join(fol_name, "labels.npy")
    # labels = np.array(labels)
    # np.save(ruta_archivo, labels)
    
    # ruta_archivo = os.path.join(fol_name, "cost_list.npy")
    # cost_list = np.array(cost_list)
    # np.save(ruta_archivo, cost_list)

ruta_archivo = os.path.join(fol_name, "initial_final_angles_cost.npy")
initial_final_angles_cost = np.array(initial_final_angles_cost)
np.save(ruta_archivo, initial_final_angles_cost)
########################## Close ##############################################

lib_pax.TLPAX_reset(instrumentHandle)
lib_pax.TLPAX_close(instrumentHandle)
print("Connection to PAX1000 closed.")

rotadores.close()
print("Connection to Zaber rotators closed.")

for i in range(len(layers)):
    # print(layers[i])
    device = device_id[layers[i]]
    lib.close_device(byref(cast(device, POINTER(c_int))))
    print("Connection to Standa rotators closed.")