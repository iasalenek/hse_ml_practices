import numpy as np

from sklearn.cluster import KMeans as true_KMeans
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll

from src.kmeans import KMeans

import pytest


def data_generator(n):
    for i in range(n):
        X_1, true_labels = make_blobs(
            400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]]
        )

        yield X_1, true_labels


@pytest.mark.parametrize("inits", ["k-means++", "random"])
@pytest.mark.parametrize("X_1, true_labels", data_generator(500))
def test_Kmeans(inits, X_1, true_labels):

    my_kmeans = KMeans(n_clusters=4, init=inits)
    my_kmeans.fit(X_1)
    my_labels = my_kmeans.predict(X_1)

    true_kmeans = true_KMeans(n_clusters=4, init=inits)
    true_kmeans.fit(X_1)
    true_labels = true_kmeans.predict(X_1)

    def ceck_equality(my_labels, true_labels):
        true_classified = 0
        for cluster in np.unique(true_labels):
            # if my_labels[true_labels == cluster].var() != 0.0:
            #    return False
            values, counts = np.unique(
                my_labels[true_labels == cluster], return_counts=True
            )
            true_classified += max(counts)
        if true_classified / len(true_labels) < 0.75:
            return False
        return True

    assert ceck_equality(my_labels, true_labels)
