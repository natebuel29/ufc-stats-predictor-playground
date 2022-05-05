import numpy as np
from predictor_playground.logistic_regression import LogisticRegression
from predictor_playground.logistic_regression_functions import costFunction, plotChart
import unittest
from scipy.optimize import minimize
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
        testdata = np.loadtxt(
            "C:\\Users\\nateb\\Documents\\Repos\\python\\ufc-stats-predictor-playground\\tests\\testdata.csv", delimiter=",").astype("float")
        rows, columns = testdata.shape
        theta = np.array([0, 0, 0])
        y = testdata[:, columns-1]
        X = np.concatenate([np.ones((rows, 1)),
                            testdata[:, 0:columns-1]], axis=1)
        logReg = LogisticRegression(X, y, theta)
        iteratations = 1000000
        j_history = logReg.gradient_descent(0.001, iteratations)
        res = minimize(
            costFunction,  np.array([0, 0, 0]), (X, y), jac=True, method='TNC', options={'maxiter': 800})
        print(f"Theta generated by scipy minmimize: {res.x}")
        print(f"Theta generated by gradient descent: {logReg.theta}")
        plotChart(iteratations, j_history)
