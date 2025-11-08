from centroid import centroide
from density_matrix import density_matrix_function
import numpy as np
from distance import distance_function

# Initialize the quantum circuit to be simulated

def test(quantum_states, labels):
    fidelities = np.matmul(quantum_states,labels)
    fidelities = fidelities*np.conj(fidelities)
    first_label = fidelities[0]
    second_label = fidelities[1]
    max_fidelities = [np.max([first_label[i], second_label[i]]) for i in range(len(fidelities[0]))]
    arg_fidelities = [np.argmax([first_label[i], second_label[i]]) for i in range(len(fidelities[0]))]

    return max_fidelities, arg_fidelities

def cost_function(quantum_states, coordinates, labels):
    # Compute prediction for each input in data batch

    loss = 0  # initialize loss

    fidelities, arg_fidelities = test(quantum_states,labels)

    coordinates = np.array(coordinates)

    arg_fidelities = np.array(arg_fidelities)

    for i in range(len(coordinates)):

        f_i = fidelities[i]

        for j in range(i + 1, (len(coordinates))):

            f_j = fidelities[j]

            if arg_fidelities[i] == arg_fidelities[j]:

                delta_y = 1
            else:

                delta_y = 0

            loss = loss + delta_y * (4*distance_function(coordinates[i], coordinates[j]) ** 2 + distance_function(coordinates[i], centroide(coordinates[np.where(arg_fidelities == arg_fidelities[i])]))) * ((1 - f_i**2) * (1 - f_j**2))

    return np.real(loss / len(coordinates))