import numpy as np

class LinearRegression:

    def __init__(self, lr: float=0.01, n_iter: int=10000):
        self.lr = lr
        self.n_iter = n_iter
        self.parameters = {}

    def initialize_parameters(self, shape: int):
        self.parameters['W'] = np.random.randn(shape) * 0.01
        self.parameters['b'] = 0.0


    def compute_cost_and_gradients(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
        m = X_train.shape[0]
        y_hat = np.dot(X_train, self.parameters['W']) + self.parameters['b']
        error = y_hat - y_train

        cost = (1 / (2 * m)) * np.sum(error ** 2)
        dJ_dW = (1 / m) * np.dot(X_train.T, error)
        dJ_db = (1 / m) * np.sum(error)

        return cost, dJ_dW, dJ_db


    def fit(self, X_train: np.ndarray, y_train: np.ndarray):

        self.initialize_parameters(X_train.shape[1])

        for epoch in range(self.n_iter):

            cost, dJ_dW, dJ_db = self.compute_cost_and_gradients(X_train, y_train)
            self.parameters['W'] = self.parameters['W'] - self.lr * dJ_dW
            self.parameters['b'] = self.parameters['b'] - self.lr * dJ_db


            if (epoch) % 1000 == 0:
                print(f'Iteration: {epoch} | Cost: {cost:.3f}')

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.dot(X_test, self.parameters['W']) + self.parameters['b']
