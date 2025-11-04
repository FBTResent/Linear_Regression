import numpy as np
from tqdm import trange
class LogisticRegression:
    """
    Logistic Regression:
    X: mxn
    y: mx1
    """

    def __init__(self):
        self.w = None          # nx1
        self.w_history = []    # list of nx1
        self.cost = 0
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def concate(self, X):
        """Add bias column"""
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def cost_function(self, X, y):
        """
        X: mxn, y: mx1, w: nx1
        """
        m = X.shape[0]
        h = self.sigmoid(X.dot(self.w))  # mx1
        epsilon = 1e-8
        cost = (1/m) * (-y.T.dot(np.log(h + epsilon)) - (1 - y).T.dot(np.log(1 - h + epsilon)))
        return cost.item()

    def gradient(self, X, y):
        """
        Compute gradient: nx1
        """
        m = X.shape[0]
        h = self.sigmoid(X.dot(self.w))  # mx1
        grad = (1/m) * X.T.dot(h - y)    # nx1
        return grad

    def gradient_descend(self, X, y, learning_rate):
        self.w = self.w - learning_rate * self.gradient(X, y)  # nx1

    def Mini_Batch_GD(self, X, y, learning_rate, batches):
        """
        Mini-batch gradient descent
        X: mxn, y: mx1
        """
        m = X.shape[0]
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        X_split = np.array_split(X_shuffled, batches)
        y_split = np.array_split(y_shuffled, batches)

        for Xi, yi in zip(X_split, y_split):
            self.gradient_descend(Xi, yi, learning_rate)
            
        
    def SGD(self, X, y, learning_rate):
        """
        Stochastic gradient descent
        X: mxn, y: mx1
        """
        m = X.shape[0]
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(m):
            Xi = X_shuffled[i].reshape(1, -1)  # 1xn
            yi = y_shuffled[i].reshape(1, -1)  # 1x1
            self.gradient_descend(Xi, yi, learning_rate)

    def fit(self, X, y, learning_rate=1e-4, epochs=10000, batches=20):
        """
        X: mxn, y: mx1
        """
        X = self.concate(X)
        if y.ndim == 1:
            y = y.reshape(-1,1)
        self.w = np.random.randn(X.shape[1], 1) * 0.01  # nx1

        for _ in trange(epochs, desc="Epochs"):
            self.Mini_Batch_GD(X, y, learning_rate, batches)
            # self.SGD(X, y, learning_rate)
            self.w_history.append(self.w.copy())
            self.cost_history.append(self.cost_function(X, y))
        self.best_w()

    def predict(self, X, threshold=0.5):
        X = self.concate(X)
        z = self.sigmoid(X.dot(self.w))
        y_pred = np.where(z >= threshold, 1, 0)
        return y_pred

    def best_w(self):
        min_index = np.argmin(self.cost_history)
        self.w = self.w_history[min_index]
        self.cost = self.cost_history[min_index]

    def __str__(self):
        return "Logistic Regression Model"
