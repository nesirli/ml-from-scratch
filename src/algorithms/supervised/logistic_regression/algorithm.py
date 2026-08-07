import numpy as np

class LogisticRegression:
	def __init__(self, n_iter: int = 3000, lr: float = 0.01):
		self.n_iter = n_iter
		self.lr = lr 
		self.threshold = 0.5
		self.parameters = {}

	def initialize_parameters(self, feature_count: int):
		self.parameters['W'] = np.random.randn(feature_count) * 0.01
		self.parameters['b'] = 0.0

	def sigmoid(self, x: np.ndarray) -> np.ndarray:
		return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


	def compute_cost_and_gradients(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
		m = X_train.shape[0]

		preds = self.predict_proba(X_train)
		error = preds - y_train
		preds = np.clip(preds, 1e-15, 1 - 1e-15)
		
		cost = - (1 / m) * np.sum(y_train * np.log(preds) + (1 - y_train) * np.log(1 - preds))
		dJdW = (1 / m) * np.dot(X_train.T, error)
		dJdb = (1 / m) * np.sum(error)

		return cost, dJdW, dJdb

	def fit(self, X_train: np.ndarray, y_train: np.ndarray):
		self.initialize_parameters(X_train.shape[1])

		for epoch in range(self.n_iter):

			cost, dJdW, dJdb = self.compute_cost_and_gradients(X_train, y_train)

			self.parameters['W'] = self.parameters['W'] - self.lr * dJdW
			self.parameters['b'] = self.parameters['b'] - self.lr * dJdb

			if epoch % 200 == 0:
				print(f'Epoch: {epoch} | Cost: {cost:.3f}')

	def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
		return self.sigmoid(np.dot(X_test, self.parameters['W']) + self.parameters['b'])

	def predict(self, X_test: np.ndarray) -> np.ndarray:
		preds_proba = self.predict_proba(X_test)
		return (preds_proba > self.threshold).astype(int)