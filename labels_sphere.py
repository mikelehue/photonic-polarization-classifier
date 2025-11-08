
from fibonacci_sphere import fibonacci_sphere_function
from cart_to_coef import cart_to_coef_function
import numpy as np

def state_labels(number_labels):

    labels = []

    # To place labels on the sphere we make use of the function fibonacci, which approximates an even distribution of points on a sphere (better performance with higher number of points)

    points_x_labels, points_y_labels, points_z_labels = fibonacci_sphere_function(number_labels)

    for i in range(len(points_x_labels)):
        c1_round, c2_round = cart_to_coef_function(points_x_labels[i], points_y_labels[i], points_z_labels[i])

        labels.append([[c1_round], [c2_round]])

    state_labels = np.array(labels, dtype=complex)


    if number_labels == 2:
        points_x_labels = [0,0]
        points_y_labels = [0, 0]
        points_z_labels = [1,-1]
        c1, c2 = cart_to_coef_function(points_x_labels,points_y_labels,points_z_labels)
        label_0 = [[c1[0]], [c2[0]]]
        label_1 = [[c1[1]], [c2[1]]]
        state_labels = np.array([label_0, label_1], dtype= complex)
 
    elif number_labels == 3 :
        labels=[]
        points_x_labels = [0,0.866, -0.866]
        points_y_labels = [0, 0, 0]
        points_z_labels = [1, -0.5, -0.5]
        for i in range(len(points_x_labels)):
 
          c1_round, c2_round = cart_to_coef_function(points_x_labels[i],points_y_labels[i],points_z_labels[i])
          labels.append([[c1_round], [c2_round]])
 
        state_labels = np.array(labels, dtype= complex)
 
    elif number_labels == 4:
        points_x_labels = [0,  0.943, -0.471, -0.471]
        points_y_labels = [0, 0, 0.816, -0.816]
        points_z_labels = [1, -0.333, -0.333, -0.333]
        c1, c2 = cart_to_coef_function(points_x_labels,points_y_labels,points_z_labels)
        label_0 = [[c1[0]], [c2[0]]]
        label_1 = [[c1[1]], [c2[1]]]
        label_2 = [[c1[2]], [c2[2]]]
        label_3 = [[c1[3]], [c2[3]]]
        state_labels = np.array([label_0, label_1, label_2, label_3], dtype= complex)

    elif number_labels == 6:
        label_0 = [[1], [0]]
        label_1 = [[0], [1]]
        label_2 = [[1/np.sqrt(2)],[1/np.sqrt(2)]]
        label_3 = [[1/np.sqrt(2)],[-1/np.sqrt(2)]]
        label_4= [[1/np.sqrt(2)],[complex(0,1)/np.sqrt(2)]]
        label_5= [[1/np.sqrt(2)],[-complex(0,1)/np.sqrt(2)]]
        state_labels = np.array([label_0, label_1, label_2, label_3, label_4, label_5], dtype= complex)


    return state_labels
