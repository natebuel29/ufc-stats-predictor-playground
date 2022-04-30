import numpy as np
from sklearn.preprocessing import normalize


class LogisticRegression:
    def __init__(self, X, y, theta):
        # standardize the data
        # def standardize(X_tr):
        # for i in range(shape(X_tr)[1]):
        # X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])
        # https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
        # https://github.com/drbilo/logistic-regression
        self.X = X
        self.y = y
        self.theta = theta

    def sigmoid(self, h):
        return (1.0/(1.0+np.exp(-h)))

    def h_of_x(self):
        return self.sigmoid(self.X.dot(np.transpose(self.theta)))

    def cost(self):
        # this is working but would like to test it more
        m = self.y.size
        h = self.h_of_x()
        first_part = (-self.y).dot(np.log(h))
        print(first_part)
        return (1/m)*((-self.y).dot(np.log(h)) - (1-self.y).dot(np.log(1-h)))
