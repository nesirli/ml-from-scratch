import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class LogisticRegression:

    def __init__(self, n_iter: int = 3000, lr: float = 0.01):
        self.n_iter = n_iter
        self.lr = lr
        self.threshold = 0.5
        self.parameters = {}

    def initialize_parameters(self, feature_count: int):
        self.parameters["W"] = np.random.randn(feature_count) * 0.01
        self.parameters["b"] = 0.0

    def compute_cost_and_gradients(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
        m = X_train.shape[0]

        preds = self.predict_proba(X_train)
        error = preds - y_train
        preds = np.clip(preds, 1e-15, 1 - 1e-15)

        # Binary cross-entropy cost
        cost = -(1 / m) * np.sum(
            y_train * np.log(preds) + (1 - y_train) * np.log(1 - preds)
        )
        # Gradients match the linear regression form
        dJdW = (1 / m) * np.dot(X_train.T, error)
        dJdb = (1 / m) * np.sum(error)

        return cost, dJdW, dJdb

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.initialize_parameters(X_train.shape[1])

        for epoch in range(self.n_iter):
            cost, dJdW, dJdb = self.compute_cost_and_gradients(X_train, y_train)

            # Gradient descent update
            self.parameters["W"] = self.parameters["W"] - self.lr * dJdW
            self.parameters["b"] = self.parameters["b"] - self.lr * dJdb

            if epoch % 200 == 0:
                print(f"Epoch: {epoch} | Cost: {cost:.3f}")

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return sigmoid(np.dot(X_test, self.parameters["W"]) + self.parameters["b"])

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # Classify with a 0.5 threshold
        preds_proba = self.predict_proba(X_test)
        return (preds_proba > self.threshold).astype(int)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression as sklearn_LogReg
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)

    cancer_data = load_breast_cancer(as_frame=True)

    X = cancer_data.data
    y = cancer_data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    custom_model = LogisticRegression()
    custom_model.fit(X_train_scaled, y_train)
    y_pred_proba = custom_model.predict_proba(X_test_scaled)
    y_pred = custom_model.predict(X_test_scaled)

    print(f"ROC AUC score of the custom model: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print(f"F1 Score of the custom model: {f1_score(y_test, y_pred):.3f}")

    sklearn_model = sklearn_LogReg()
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred_proba = sklearn_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = sklearn_model.predict(X_test_scaled)

    print(f"ROC AUC score of the sklearn model: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print(f"F1 Score of the sklearn model: {f1_score(y_test, y_pred):.3f}")
