import pandas as pd
import numpy as np
import math
class LogisticRegression:
    """
    Logistic Regression: class
    -----
        Logistic regression make by me :3
    ----
    """

    def __init__(self):
        self.w_history = []
        self.cost_history=[]
        self.w = []
        self.cost = 0

    """Hàm sigmoid"""
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    

    """Hàm lost"""
    def cost_function(self, X, y):
        m=X.shape[0]
        epsilon = 1e-8
        h=self.sigmoid(X.dot(self.w))
        cost = (1/m)*(-y.T.dot(np.log(h+epsilon))-(1-y).T.dot(np.log(1-h+epsilon)))
        return cost.item()
    

    """gradient"""
    def gradient(self, X, y):
        m = X.shape[0]
        h = self.sigmoid(X.dot(self.w))
        grad = (1/m)*X.T.dot(h-y)
        return grad
    
    def gradient_descend(self, X, y, learning_rate):
        self.w = self.w - learning_rate * self.gradient(X, y)
        self.w_history.append(self.w)
        self.cost_history.append(self.cost_function(X,y))

    def SGD(self, X, y, learning_rate, batches):
        indices = np.random.permutation(X.shape[0])
        X_split = np.split(X[indices],batches)
        y_split = np.split(y[indices],batches)
        for Xi,yi in zip(X_split,y_split):
                self.gradient_descend(Xi, yi, learning_rate)
        
    def concate(self,X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return X

    def fit(self, X, y, learning_rate=1e-4, epochs=10000, batches=20):
        """
        Parameter
        ----------
            X: np.ndarray of shape(n_sample, n_feature)
            Input feature.\n
            y: np.ndarray of shape(n_sample, 1)
            Input label.\n
            learning_rate: float, default=1e-2
            Step size update weight.\n
            epochs: int, default=1000
            Number of training iteration.\n
            batches: int, default=20
            Number of mini-batches per epoch.\n
        
        Return
        ------
        None
        """
        X = self.concate(X)
        self.w = np.random.randn(X.shape[1]) * 0.01 #np.random.randint(1,2,(X.shape[1]))
        for _ in range(int(epochs)):
            #self.SGD(X, y, learning_rate, batches)
            self.gradient_descend(X,y,learning_rate)
        self.best_w()

    
    def predict(self, X, threshold=0.5):
        X = self.concate(X)
        z = self.sigmoid(X.dot(self.w))     # Xác suất class 1
        y_pred = np.where(z >= threshold, 1, 0)  # Dùng threshold xác suất
        return y_pred
    
    def __str__(self):
        pass
    def best_w(self):
        # print(np.asarray(self.cost_history).shape,np.asarray(self.w_history).shape)
        min_index = np.argmin(self.cost_history)
        self.cost = self.cost_history[min_index]
        self.w = self.w_history[min_index]
    
    