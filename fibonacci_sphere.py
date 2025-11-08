import math

def fibonacci_sphere_function(samples):
    points_x = []
    points_y = []
    points_z = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points_x.append(x)
        points_y.append(y)
        points_z.append(z)
    return points_x, points_y, points_z