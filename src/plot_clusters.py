import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll

from cluster_visualization import visualize_clasters, clusters_statistics
from kmeans import KMeans


def plot_clusters():
    X_1, true_labels = make_blobs(
        400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]]
    )

    X_2, true_labels = make_moons(400, noise=0.075)

    kmeans = KMeans(n_clusters=4, init="k-means++")
    kmeans.fit(X_1)
    labels = kmeans.predict(X_1)
    visualize_clasters(X_1, labels)

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X_2)
    labels = kmeans.predict(X_2)
    visualize_clasters(X_2, labels)
