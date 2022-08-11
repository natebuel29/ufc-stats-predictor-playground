import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from logistic_regression.logistic_regression_functions import *

from util import construct_data
from sklearn.feature_selection import RFE
from scipy.optimize import minimize


def log_reg_ufc_test():
    X, y, X_test, y_test, X_future = construct_data()

    rows, columns = X.shape
    X = np.concatenate([np.ones((rows, 1)),
                        X], axis=1)
    X_future = np.concatenate([np.ones((X_future.shape[0], 1)),
                               X_future], axis=1)
    X_test = np.concatenate([np.ones((X_test.shape[0], 1)),
                             X_test], axis=1)

    clf = LogisticRegression(random_state=2)
    # Use Recursive Feature Elimation for feature selection
    rfe = RFE(clf)
    fit = rfe.fit(X, y)
    X = X[:, fit.support_]
    X_test = X_test[:, fit.support_]

    rows, columns = X.shape
    X = np.concatenate([np.ones((rows, 1)),
                        X], axis=1)
    X_future = X_future[:, fit.support_]
    X_future = np.concatenate([np.ones((X_future.shape[0], 1)),
                               X_future], axis=1)
    X_test = np.concatenate([np.ones((X_test.shape[0], 1)),
                            X_test], axis=1)
    start_theta = np.zeros(columns+1)

    print("------------Logistic Regression---------------")
    #print(f"test results for  {correct/total}")

    # use the scipy.optimize.minimize function to train our parameters
    # compared the generated weights and they matched my gradient descent function (:
    res = minimize(costFunction, start_theta, (X, y),
                   jac=True, method="BFGS", options={'maxiter': 400})
    clf.fit(X, y)
    clf_predictions = clf.predict(X_future)
    print("predictions for a future fight card")
    print(clf.predict(X_future))
    print("Score on test data")
    print(clf.score(X_test, y_test))
    print("------------Custom Logistic Regresion---------------")
    print("predictions for a future fight card")
    lg_results = predict_X(res.x, X_future)
    print(lg_results)


def main():
    log_reg_ufc_test()


if __name__ == "__main__":
    main()
