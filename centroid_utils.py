import numpy as np

def centroide (x):
    centroid = []
    centroid= sum(x)
    centroid= centroid /len(x)
    return np.array(centroid)
