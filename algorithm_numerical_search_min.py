# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:00:56 2023

@author: QNanoLab1
"""
import numpy as np
import matplotlib.pyplot as plt

import os
from cost import cost_function
from Labels_sphere import state_labels
from numpy import cos as cos
from numpy import sin as sin
from numpy import pi as pi
import matplotlib.image as mpimg

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

# Folder for saving the data
# fol_name = input("Folder name: ")
fol_name = 'data'
if not os.path.exists(fol_name):
    os.mkdir(fol_name)

def test(quantum_states, labels):
    fidelities = np.abs(np.matmul(quantum_states, labels.conj().T))**2
    max_fidelities = np.max(fidelities, axis=2)
    arg_fidelities = np.argmax(fidelities, axis=2)

    return max_fidelities, arg_fidelities

############# DATA FOR THE ALGORITHM ##########################################

#Choose number of labels
number_labels = 4
labels = state_labels(number_labels)
 
#Choose number of gates
number_gates = 2  #Initialization gates do not count

#Choose the number of times the whole set of gates is applied
number_iterations = 50

#Choose the step for calculate the gradient
st = 0.01

#Choose the value of the learning rate
lr = 0.006

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
coordinates_jones = []

# Generate points within clusters
for i in range(number_clusters):
    # Generate points within current cluster
    for j in range(points_per_cluster):
        # Generate point with Gaussian distribution
        point = np.random.normal(loc=centers[i], scale=widths[i])
        coordinates_cartesian.append([point[0], point[1], 0])
        coordinates_jones.append(Stokes2Jones(point[0],point[1]))

################### Algorithm #################################################

coordinates_cartesian= np.array(coordinates_cartesian)

#write matrixes that will represent quarter and half
quarter = lambda th: np.array([[cos(th)**2+complex(0,1)*sin(th)**2, (1-complex(0,1))*sin(th)*cos(th)], [(1-complex(0,1))*sin(th)*cos(th),sin(th)**2+complex(0,1)*cos(th)**2]])
half = lambda th: np.array([[cos(th)**2-sin(th)**2, 2*sin(th)*cos(th)], [2*sin(th)*cos(th),sin(th)**2-cos(th)**2]])

# Cargar la imagen
imagen = mpimg.imread('cost_4_labels.png')

# Crear una figura y un eje para el gráfico
fig, ax = plt.subplots()

# Trazar la imagen de fondo
ax.imshow(imagen, extent=[0, 2*np.pi, 0, 2*np.pi], aspect='auto')

# #Randomly initialize the angles of the gates

colores = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

N_rec = 10

final_cost_list = []
final_angles = []

for i_rec in range(N_rec):
    
    print('recorrido No. ', i_rec)

    params = np.random.uniform(size=(number_gates))*2*np.pi
    
    #Initial state
    Initial_state =np.array([1,0])
    
    quantum_states_list = []
    quantum_state = []
    quantum_states = []
    
    # layer_values = np.dot(half(params[3]),np.dot(quarter(params[2]),np.dot(half(params[1]),quarter(params[0]))))
    
    layer_values = np.eye(2)
    
    for i in range(len(params)):
        if i % 2 == 0:
            layer_values = np.dot(quarter(params[i]), layer_values)
        else:
            layer_values = np.dot(quarter(params[i]), layer_values)
    
    Initialization_list = coordinates_jones
    
    for i in range(len(coordinates_cartesian)):
        quantum_state = np.dot(layer_values,Initialization_list[i]) 
        quantum_states.append(quantum_state)
        
        
    quantum_states_list.append(quantum_states)
        
    # Save first data
    
    cost_value = cost_function(quantum_states,coordinates_cartesian,labels)
    
    costf = cost_value
    cost_list = []
    max_fidel, max_arg = test(quantum_states, labels)
    
    max_arg_list = []
    max_arg_list.append(max_arg)
    
    #Initialize gradient descent
    Gd = np.zeros(len(params))
    
    # print('End of first step of the algorithm.')
    
    angles = []
    
    #Start iterative procedure. As many epochs as number_iterations
    for index in range(number_iterations):
        
        # print('*************************************************')
        
        #Introduce the parameters of the rotation layers
        params_initial = params.copy()
        x = params.copy()
    
        angles.append(x)
    
        #Create artificial gradient descent
        for j in range(len(x)):
    
            # print(index, 'mini epoch layer number:  ', j)
    
            x[j] += st
    
            layer_values = np.eye(2)
    
            for i in range(len(params)):
                if i % 2 == 0:
                    layer_values = np.dot(quarter(x[i]), layer_values)
                else:
                    layer_values = np.dot(quarter(x[i]), layer_values)
    
            # layer_values  = np.dot(half(x[3]),np.dot(quarter(x[2]),np.dot(half(x[1]),quarter(x[0]))))
    
            
            quantum_states = []
            
            for i in range(len(coordinates_cartesian)):
                
                  quantum_state = np.dot(layer_values,Initialization_list[i])
                 
                  quantum_states.append(quantum_state)
            
            #we move all the itermediate layers before initializing the points
    
            cost_value = cost_function(quantum_states,coordinates_cartesian,labels)
    
            # print('cost_value = ', cost_value)
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
    
        # layer_values = np.dot(half(params[3]),np.dot(quarter(params[2]),np.dot(half(params[1]),quarter(params[0]))))
    
        layer_values = np.eye(2)
    
        for i in range(len(params)):
            if i % 2 == 0:
                layer_values = np.dot(quarter(params[i]), layer_values)
            else:
                layer_values = np.dot(quarter(params[i]), layer_values)
    
        quantum_states =[]
        
        for i in range(len(coordinates_cartesian)):
                
                  quantum_state = np.dot(layer_values,Initialization_list[i])
    
                  quantum_states.append(quantum_state)
                 
        max_fidel, max_arg = test(quantum_states, labels)
        # print('quantum_states = ', quantum_states)
        # print('labels = ', labels)
        # print('max_fidel = ', max_fidel)
        # print('max_arg = ', max_arg)
    
        costf = cost_function(quantum_states,coordinates_cartesian,labels)
    
        # print('costf = ', costf)
    
        cost_list.append(costf)
        
        quantum_states_list.append(quantum_states)
        max_arg_list.append(max_arg)

    final_cost_list.append(costf)
    final_angles.append(params)


    angles = np.array(angles)
    angles = angles % (2 * np.pi)

    # Separa las coordenadas x e y en listas separadas para facilitar el trazado
    x_coords, y_coords = zip(*angles)
    
    # Trazar el recorrido
    ax.plot(x_coords, y_coords, marker='.', linestyle=' ', color='red')


final_angles_cost = np.hstack((final_angles, np.array(final_cost_list).reshape(-1, 1)))

# Configurar etiquetas y títulos
ax.set_xlabel('Half angle')
ax.set_ylabel('Quarter angle')
ax.set_title('Cost function')
ax.grid(True)

# Mostrar el gráfico
plt.show()



# ruta_archivo = os.path.join(fol_name, "quantum_states.npy")
# quantum_states_list = np.array(quantum_states_list)
# np.save(ruta_archivo, quantum_states_list)

# ruta_archivo = os.path.join(fol_name, "max_arg.npy")
# max_arg_list = np.array(max_arg_list)
# np.save(ruta_archivo, max_arg_list)

# ruta_archivo = os.path.join(fol_name, "coordinates_theoretical.npy")
# coordinates_cartesian = np.array(coordinates_cartesian)
# np.save(ruta_archivo, coordinates_cartesian)

# ruta_archivo = os.path.join(fol_name, "labels.npy")
# labels = np.array(labels)
# np.save(ruta_archivo, labels)
