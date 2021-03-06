import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def sigmoid(h):
    return (1.0/(1.0+np.exp(-h)))


def costFunction(theta, X, y):
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


def predict(theta, x):
    pred = sigmoid(x.dot(theta))

    # if pred > 0.5 -> predict red fighter won (1)
    # if pred < 0.5 -> predict blue fighter won (2)
    return 1 if pred > 0.5 else 0


def predict_X(theta, X):
    results = []
    for i in range(0, X.shape[0]):
        x = X[i]
        result = predict(theta, x)
        results.append(result)
    return results


def standardize(X):
    X_norm = X.copy()
    mu = np.mean(X_norm, axis=0)
    sigma = np.std(X_norm, axis=0)
    X_norm = (X_norm - mu)/sigma
    return X_norm
