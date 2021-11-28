import numpy as np
from typing import NoReturn
import copy

def MinMaxScaler(X: np.array, params=None):
    if not params:
        X_min, X_max = [np.min(X, axis=0), np.max(X, axis=0)]
        return (X - X_min) / (X_max - X_min), [X_min, X_max]
    else:
        return (X - params[0]) / (params[1] - params[0])


def dist(x1, x2):
    return np.linalg.norm(x1 - x2, axis=1)


def center_of_mass(X):
    return X.mean(axis=0)


def cdist(X, Y):
    return np.array([dist(X, Y[i]) for i in range(len(Y))]).T


class KMeans:
    def __init__(self, n_clusters: int, init: str = "random", max_iter: int = 300):
        """
        
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        
        """

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.scaler = [np.nan, np.nan]

    def fit(self, X: np.array, y=None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.
        
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать 
            параметры X и y, даже если y не используется).
        
        """

        X, self.scaler = MinMaxScaler(X)

        if self.init == "random":
            clusters = np.array([])
            while len(np.unique(clusters)) < self.n_clusters:
                centers = np.random.rand(self.n_clusters, X.shape[1])
                clusters = cdist(X, centers).argmin(axis=1)

        elif self.init == "sample":
            centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
            clusters = cdist(X, centers).argmin(axis=1)

        elif self.init == "k-means++":
            centers = X[np.random.choice(X.shape[0], 1, replace=False)]

            while len(centers) != self.n_clusters:
                probability = cdist(X, centers).min(axis=1) ** 2
                probability = probability / sum(probability)
                new_center = X[
                    np.random.choice(X.shape[0], 1, replace=False, p=probability)
                ]
                centers = np.append(centers, new_center, axis=0)

            clusters = cdist(X, centers).argmin(axis=1)

        new_centres = [center_of_mass(X[clusters == i]) for i in range(self.n_clusters)]
        new_clusters = cdist(X, new_centres).argmin(axis=1)

        while sum(clusters != new_clusters) != 0:

            centers = copy.deepcopy(new_centres)
            clusters = copy.deepcopy(new_clusters)

            new_centres = [
                center_of_mass(X[clusters == i]) for i in range(self.n_clusters)
            ]
            new_clusters = cdist(X, new_centres).argmin(axis=1)

        self.centers = centers

    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера, 
        к которому относится данный элемент.
        
        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.
        
        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров 
            (по одному индексу для каждого элемента из X).
        
        """

        X = MinMaxScaler(X, self.scaler)
        return cdist(X, self.centers).argmin(axis=1)