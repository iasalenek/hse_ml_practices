import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll

def generate_blobs(n_samples=400, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]]):
    X_1, y_1 = make_blobs(
        n_samples, 2, centers
    )
    return X_1, y_1

def generate_moons(n_samples=400, noise=0.075):
    X_2, y_2 = make_moons(n_samples, noise)
    return X_2, y_2

