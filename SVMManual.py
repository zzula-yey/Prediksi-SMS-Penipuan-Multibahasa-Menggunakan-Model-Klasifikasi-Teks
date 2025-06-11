import numpy as np


class LinearSVM:
    def __init__(self, C=1, learning_rate=0.001, max_iter=1000, random_state=42):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.w = None
        self.b = 0

    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        np.random.seed(self.random_state)

        # Inisialisasi bobot
        self.w = np.zeros(n_features)
        self.b = 0

        # Training
        for _ in range(self.max_iter):
            for i in range(n_samples):
                condition = y[i] * (np.dot(X[i], self.w) + self.b) < 1
                if condition:
                    self.w -= self.learning_rate * (self.C * y[i] * X[i])
                    self.b -= self.learning_rate * self.C * y[i]
                # Tidak ada update jika kondisi margin >= 1

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, 0)