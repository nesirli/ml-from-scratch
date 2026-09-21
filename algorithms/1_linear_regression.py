import numpy as np


class LinearRegression:

    def __init__(self, lr: float = 0.01, n_iter: int = 10000):
        self.lr = lr
        self.n_iter = n_iter
        self.parameters = {}

    def initialize_parameters(self, shape: int):
        # Small random weights and a zero bias
        self.parameters["W"] = np.random.randn(shape) * 0.01
        self.parameters["b"] = 0.0

    def compute_cost_and_gradients(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
        m = X_train.shape[0]
        y_hat = np.dot(X_train, self.parameters["W"]) + self.parameters["b"]
        error = y_hat - y_train

        # Mean squared error cost
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        # Gradients of the cost with respect to W and b
        dJ_dW = (1 / m) * np.dot(X_train.T, error)
        dJ_db = (1 / m) * np.sum(error)

        return cost, dJ_dW, dJ_db

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.initialize_parameters(X_train.shape[1])

        for epoch in range(self.n_iter):
            cost, dJ_dW, dJ_db = self.compute_cost_and_gradients(X_train, y_train)
            # Gradient descent update
            self.parameters["W"] = self.parameters["W"] - self.lr * dJ_dW
            self.parameters["b"] = self.parameters["b"] - self.lr * dJ_db

            if epoch % 1000 == 0:
                print(f"Iteration: {epoch} | Cost: {cost:.3f}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.dot(X_test, self.parameters["W"]) + self.parameters["b"]


if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.linear_model import LinearRegression as sklearn_LR
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)

    housing_data = fetch_california_housing(as_frame=True)

    X = housing_data.data
    y = housing_data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    custom_model = LinearRegression()
    custom_model.fit(X_train_scaled, y_train)
    y_pred = custom_model.predict(X_test_scaled)

    print(f"Mean squared error of the custom model: {mean_squared_error(y_test, y_pred):.3f}")
    print(f"R2 Score of the custom model: {r2_score(y_test, y_pred):.3f}")

    sklearn_model = sklearn_LR()
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred = sklearn_model.predict(X_test_scaled)

    print(f"Mean squared error of the sklearn model: {mean_squared_error(y_test, y_pred):.3f}")
    print(f"R2 Score of the sklearn model: {r2_score(y_test, y_pred):.3f}")
