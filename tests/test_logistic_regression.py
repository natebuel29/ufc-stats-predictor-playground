import numpy as np
from predictor_playground.logistic_regression import LogisticRegression
import unittest


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
        # finish this test
        theta = np.array([0.2, 0.1, 0.5])
        X = np.array([[1, 1, 1], [1, 1, 1]])
        y = np.array([1.0, 0.0])
        expected = 0
        logReg = LogisticRegression(X, y, theta)
        results = logReg.cost()
        self.assertEqual(results, expected)
