from collections import Counter

import numpy as np


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Euclidean distance between two points
    return np.sqrt(np.sum((x - y) ** 2))


class KNeighborsBase:
    def __init__(self, k: int = 3):
        self.k = k

    def _get_neighbor_labels(self, x_test: np.ndarray) -> list:
        # Find the k nearest training labels
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
        # Majority vote
        most_common = Counter(labels).most_common(1)[0][0]
        return most_common

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        probas = []
        for x_test in X_test:
            labels = self._get_neighbor_labels(x_test)
            counts = Counter(labels)
            # Fraction of neighbors in each class
            probs = [counts.get(c, 0) / self.k for c in self._classes]
            probas.append(probs)
        return np.array(probas)


class KNeighborsRegressor(KNeighborsBase):
    def _predict(self, x_test: np.ndarray) -> int:
        # Mean of neighbor targets
        labels = self._get_neighbor_labels(x_test)
        return np.mean(labels)


if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing, load_breast_cancer
    from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
    from sklearn.neighbors import KNeighborsClassifier as sklearn_knn_clf
    from sklearn.neighbors import KNeighborsRegressor as sklearn_knn_reg
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)

    cancer_data = load_breast_cancer()
    housing_data = fetch_california_housing()

    # ========================================= KNN Classification =========================================

    X, y = cancer_data.data, cancer_data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    custom_knn_clf = KNeighborsClassifier(k=5)
    custom_knn_clf.fit(X_train_scaled, y_train)
    y_pred_prob = custom_knn_clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = custom_knn_clf.predict(X_test_scaled)
    print(
        f"Custom KNN classifier ROC AUC score: {roc_auc_score(y_test, y_pred_prob):.3f} | F1 score: {f1_score(y_test, y_pred):.3f}"
    )

    cls_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = []
    for train_idx, val_idx in cls_cv.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        model = KNeighborsClassifier(k=5)
        model.fit(X_tr_fold_scaled, y_train_fold)
        y_pred = model.predict_proba(X_val_fold_scaled)[:, 1]
        scores.append(roc_auc_score(y_val_fold, y_pred))

    print(f"Mean custom KNN ROC AUC: {np.mean(scores):.3f} | Std: {np.std(scores):.3f}")

    sklearn_knn_clf = sklearn_knn_clf(n_neighbors=5)
    sklearn_knn_clf.fit(X_train_scaled, y_train)
    y_pred_prob = sklearn_knn_clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = sklearn_knn_clf.predict(X_test_scaled)
    print(
        f"Sklearn KNN classifier ROC AUC score: {roc_auc_score(y_test, y_pred_prob):.3f} | F1 score: {f1_score(y_test, y_pred):.3f}"
    )

    scores = cross_val_score(
        sklearn_knn_clf, X_train_scaled, y_train, cv=cls_cv, scoring="roc_auc"
    )
    print(f"Mean sklearn KNN ROC AUC: {np.mean(scores):.3f} | Std: {np.std(scores):.3f}")

    # ========================================= KNN Regression =========================================

    X, y = housing_data.data, housing_data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    custom_knn_reg = KNeighborsRegressor(k=5)
    custom_knn_reg.fit(X_train_scaled, y_train)
    y_pred = custom_knn_reg.predict(X_test_scaled)
    print(
        f"Custom KNN regressor mean squared error score: {mean_squared_error(y_test, y_pred):.3f} | R2 score: {r2_score(y_test, y_pred):.3f}"
    )

    reg_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []
    for train_idx, val_idx in reg_cv.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        model = KNeighborsRegressor(k=5)
        model.fit(X_tr_fold_scaled, y_train_fold)
        y_pred = model.predict(X_val_fold_scaled)
        scores.append(mean_squared_error(y_val_fold, y_pred))

    print(f"Mean custom KNN mean squared error: {np.mean(scores):.3f} | Std: {np.std(scores):.3f}")

    sklearn_knn_reg = sklearn_knn_reg(n_neighbors=5)
    sklearn_knn_reg.fit(X_train_scaled, y_train)
    y_pred = sklearn_knn_reg.predict(X_test_scaled)
    print(
        f"Sklearn KNN regressor mean squared error score: {mean_squared_error(y_test, y_pred):.3f} | R2 score: {r2_score(y_test, y_pred):.3f}"
    )

    scores = cross_val_score(
        sklearn_knn_reg, X_train_scaled, y_train, cv=reg_cv, scoring="neg_mean_squared_error"
    )
    print(f"Mean sklearn KNN mean squared error: {np.mean(abs(scores)):.3f} | Std: {np.std(abs(scores)):.3f}")
