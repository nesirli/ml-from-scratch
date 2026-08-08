import numpy as np

def euclidian_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
	return np.sqrt(np.sum((x - y) ** 2))

def sigmoid(x: np.ndarray) -> np.ndarray:
	return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))