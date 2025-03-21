import numpy as np


def read_den(file: str):
    header = file[0].split()
    grid = [int(i) for i in header]
    density = []
    for line in file[1:]:
        density.extend([float(i) for i in line.split()])
    density = np.array(density).reshape(grid)
    return density
