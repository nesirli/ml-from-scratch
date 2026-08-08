from collections import Counter

import numpy as np

from algorithms.utils.functions import euclidian_distance


class KNeighborsBase:
    def __init__(self, k: int = 3):
        self.k = k

    def _get_neighbor_labels(self, x_test: np.ndarray) -> list:
        distances = [euclidian_distance(x_test, x_train) for x_train in self.X_train]
        indices = np.argsort(distances)[: self.k]
        return [self.y_train[i] for i in indices]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise TypeError("Use numpy arrays for training.")
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.array([self._predict(x_test) for x_test in X_test])


class KNeighborsClassifier(KNeighborsBase):
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        super().fit(X_train, y_train)
        self._classes = np.unique(y_train)

    def _predict(self, x_test: np.ndarray) -> int:
        labels = self._get_neighbor_labels(x_test)
        most_common = Counter(labels).most_common(1)[0][0]
        return most_common

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        probas = []
        for x_test in X_test:
            labels = self._get_neighbor_labels(x_test)
            counts = Counter(labels)
            probs = [counts.get(c, 0) / self.k for c in self._classes]
            probas.append(probs)
        return np.array(probas)


class KNeighborsRegressor(KNeighborsBase):
    def _predict(self, x_test: np.ndarray) -> int:
        labels = self._get_neighbor_labels(x_test)
        return np.mean(labels)
