import numpy as np
from predictor_playground.logistic_regression import LogisticRegression
import unittest
from scipy.optimize import minimize, fmin_tnc
import csv
import matplotlib.pyplot as plt


class TestLogisticRegression(unittest.TestCase):

    def test_sigmoid(self):
        h = np.array([-50, 10, 0, -9, 73])
        expected = np.array([0, 1, 0.5, 0, 1])
        logReg = LogisticRegression([], [], [])
        results = logReg.sigmoid(h)
        self.assertEqual(np.around(results, decimals=3).all(), expected.all())

    def test_h_of_x(self):
        theta = np.array([2, 3, 6])
        X = np.array([[7, 2, 3], [-6, -3, -4]])
        y = []
        expected = np.array([1, 0])
        logReg = LogisticRegression(X, y, theta)
        results = logReg.h_of_x()
        self.assertEqual(np.around(results, decimals=3).all(), expected.all())

    def test_cost(self):
        theta = np.array([0.2, 0.1, 0.5])
        X = np.array([[1, 1, 1], [1, 1, 1]])
        y = np.array([1.0, 0.0])
        expected = .7711
        logReg = LogisticRegression(X, y, theta)
        results = logReg.cost()
        J, gradient = costFunction(theta, X, y)
        self.assertEqual(np.around(results, decimals=4), expected)

    def test_gradient_descent(self):
        irisdata = np.loadtxt(
            "C:\\Users\\nateb\\Documents\\Repos\\python\\ufc-stats-predictor-playground\\tests\\irisdata.csv", delimiter=",", skiprows=1).astype("float")
        rows, columns = irisdata.shape
        theta = np.array([0, 0, 0])
        y = irisdata[:, columns-1]
        X = np.concatenate([np.ones((rows, 1)),
                            irisdata[:, 0:columns-1]], axis=1)
        logReg = LogisticRegression(X, y, theta)
        iteratations = 30000
        j_history = logReg.gradient_descent(0.01, iteratations)
        res = minimize(
            costFunction,  np.array([0, 0, 0]), (X, y), jac=True, method='TNC', options={'maxiter': 400})
        print(res.x)
        print(res.fun)
        # print(res.x)
        # plotChart(iteratations, j_history)

        # (logReg.theta)


def costFunction(theta, X, y):
    def sigmoid(h):
        return (1.0/(1.0+np.exp(-h)))
    m = y.size
    h = sigmoid(X.dot(np.transpose(theta)))
    print(h)
    J = (1/m)*((-y).dot(np.log(h+.0000001)) - (1-y).dot(np.log(1-h+.0000001)))
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
