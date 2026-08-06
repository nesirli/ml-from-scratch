import numpy as np

class LinearRegression:

    def __init__(self, learning_rate: float=0.001, n_iterations: int=10000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.parameters = {}

    def initialize_parameters(self, shape: int):
        self.parameters['W'] = np.random.randn(shape) * 0.01
        self.parameters['b'] = 0.0

    def compute_cost(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        m = X_train.shape[0]

        y_hat = np.dot(X_train, self.parameters['W']) + self.parameters['b']
        error = y_hat - y_train

        cost = (1 / (2 * m)) * np.sum(error ** 2)
        return cost

    def compute_gradients(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
        m = X_train.shape[0]
        y_hat = np.dot(X_train, self.parameters['W']) + self.parameters['b']
        error = y_hat - y_train

        dJ_dW = (1 / m) * np.dot(X_train.T, error)
        dJ_db = (1 / m) * np.sum(error)

        return dJ_dW, dJ_db


    def fit(self, X_train: np.ndarray, y_train: np.ndarray):

        self.initialize_parameters(X_train.shape[1])

        for epoch in range(self.n_iterations):

            dJ_dW, dJ_db = self.compute_gradients(X_train, y_train)
            self.parameters['W'] = self.parameters['W'] - self.learning_rate * dJ_dW
            self.parameters['b'] = self.parameters['b'] - self.learning_rate * dJ_db

            cost = self.compute_cost(X_train, y_train)

            if (epoch + 1) % 1000 == 0:
                print(f'Iteration: {epoch+1} | Cost: {cost:.3f}')

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.dot(X_test, self.parameters['W']) + self.parameters['b']
