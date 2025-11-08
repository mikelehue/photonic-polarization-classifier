import numpy as np

def distance_function(x_1,x_2):
    return float(np.sqrt((float(x_1[0])-float(x_2[0]))**2+(float(x_1[1])-float(x_2[1]))**2+(float(x_1[2])-float(x_2[2]))**2))

