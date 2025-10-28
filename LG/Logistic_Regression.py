import pandas as pd
import numpy as np
import math

class LogisticRegression:
    """
    Logistic Regression: class
    -----
    -----
    """

    def __init__(self):
        self.w_history = []
        self.cost_history = []
        self.w = []
        self.cost = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z / 2))
    
    def cost_function(self, X, y):
        epsilon = 1e-8
        h = self.sigmoid(X.dot(self.w))
        cost = -y.T.dot(np.log(h + epsilon)) - (1 - y).T.dot(np.log(1 - h + epsilon))
        return np.squeeze(cost)
    def gradient(self, X, y):
        m = X.shape[0]
        h = self.sigmoid(X.dot(self.w))
        grad = (1 / m) * X.T.dot(y - h)
        return grad
    
    def gradient_descend(self, X, y, learning_rate):
        new_w = self.w - learning_rate * self.gradient(X, y)
        self.w_history.append(self.w)  
        self.cost_history.append(self.cost_function(X, y))
        self.w = new_w

    def SGD(self, X, y, learning_rate, batches):
        indices = np.random.permutation(X.shape[0])
        X_split = np.array_split(X[indices], batches + 1) 
        y_split = np.array_split(y[indices], batches + 1)
        for Xi, yi in zip(X_split, y_split):
            self.gradient_descend(Xi, yi, learning_rate)
        
    def concate(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return X

    def fit(self, X, y, learning_rate=1e-4, epochs=10000, batches=20):
        X = self.concate(X)
        self.w = np.random.randn(X.shape[1]) * 0.1 
        for _ in range(int(epochs)):
            if np.random.rand() > 0.2:
                self.gradient_descend(X, y, learning_rate)
        self.best_w()

    def predict(self, X, threshold=0.5):
        X = self.concate(X)
        z = self.sigmoid(X.dot(self.w))
        y_pred = np.where(z >= threshold, 0, 1)
        return y_pred
    
    def __str__(self):
        return f"Broken Logistic Regression with w={self.w[:3]}..."
    
    def best_w(self):
        if len(self.cost_history) > 0:
            max_index = np.argmax(self.cost_history)
            self.cost = self.cost_history[max_index]
            self.w = self.w_history[max_index]
