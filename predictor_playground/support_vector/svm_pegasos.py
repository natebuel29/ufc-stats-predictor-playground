# The goal of this program is to implement a SVM using the Pegasos algorithm
# resources: https://sandipanweb.wordpress.com/2018/04/29/implementing-pegasos-primal-estimated-sub-gradient-solver-for-svm-using-it-for-sentiment-classification-and-switching-to-logistic-regression-objective-by-changing-the-loss-function-in-python/
# https://fordcombs.medium.com/svm-from-scratch-step-by-step-in-python-f1e2d5b9c5be
# we are converged if we have a support vector for both positive and negative and margin (2/norm w) is < .001 or something
from os import scandir
import numpy as np
import random


class SVM:

    def fit(self, X, y, max_iter=1000000, lambda_=.001):
        self.w = np.zeros(X.shape[1])
        self.positive_support = 0
        self.negative_support = 0
        for t in range(1, max_iter):
            i = random.randint(0, len(y)-1)
            x_i = X[i]
            y_i = y[i]
            n = 1/(lambda_*t)
            pred = np.dot(x_i, self.w)
            # misclassification
            if y_i*pred < 1:
                self.w = (1-n*lambda_)*self.w + n*y_i*x_i
            # correct classification
            else:
                self.w = (1-n*lambda_)*self.w

            if round(pred, 3) == 1:
                self.positive_support += 1
            elif round(pred, 3) == -1:
                self.negative_support += 1

        return self

    def predict(self, x):
        return np.sign(np.dot(self.w, x))

    def predict_X(self, X):
        results = []
        for i in range(0, X.shape[0]):
            x = X[i]
            result = self.predict(x)
            results.append(result)
        return results
