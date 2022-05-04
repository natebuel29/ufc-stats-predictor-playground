import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def costFunction(theta, X, y):
    def sigmoid(h):
        return (1.0/(1.0+np.exp(-h)))
    m = y.size
    h = sigmoid(X.dot(np.transpose(theta)))
    J = (1/m)*((-y).dot(np.log(h)) - (1-y).dot(np.log(1-h)))
    grad = ((h-y).dot(X))/m

    return J, grad


def plotChart(iterations, cost_num):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost_num, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Iterations')
    plt.style.use('fivethirtyeight')
    plt.show()


def standardize(X):
    X_norm = X.copy()
    mu = np.mean(X_norm, axis=0)
    sigma = np.std(X_norm, axis=0)
    X_norm = (X_norm - mu)/sigma
    return X_norm
