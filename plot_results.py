# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:11:24 2023

@author: QNanoLab1
"""
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:53:18 2023

@author: QNanoLab1
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from IPython.display import display
import ipywidgets as widgets

# folder_path = input("Data folder: ")
folder_path = 'data_4_03_momento/st=0.0010, lr=0.0027'

# Cargar los datos desde la carpeta especificada
coordinates_measured = np.load(f'{folder_path}/coordinates_theoretical.npy') / np.pi
# coordinates_measured = coordinates_measured[0]
max_arg = np.load(f'{folder_path}/max_arg.npy')[-100:, :, :]
cost_list = np.load(f'{folder_path}/cost_list.npy')[-100:]

# Creamos la figura y los ejes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Creamos el slider
iteration_slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
iteration_slider = Slider(iteration_slider_ax, 'Iteration', 0, len(max_arg) - 1, valinit=0, valstep=1)



# Funciones para actualizar el plot
def update(val):
    iteration = int(iteration_slider.val)
    plot_points(iteration)
    plot_cost(iteration)
    
def plot_points(iteration):
    ax1.clear()
    for i, point in enumerate(coordinates_measured):
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'black']
        color = colors[int(max_arg[iteration][0][i])]
        ax1.scatter(point[0], point[1], color=color, marker='.')
    ax1.set_xlim([-0.5, 0.5])
    ax1.set_ylim([-0.25, 0.25])
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'$\psi$ [units of $\pi$]')
    ax1.set_ylabel(r'$\chi$ [units of $\pi$]')
    ax1.set_title('Measured Coordinates (Iteration {})'.format(iteration))

def plot_cost(iteration):
    ax2.clear()
    ax2.plot(cost_list)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.set_title('Cost Function')
    ax2.scatter(iteration, cost_list[iteration], color='red', marker='o')

# Actualizamos el plot al mover la barra
iteration_slider.on_changed(update)

# Mostramos la figura
display(fig)
