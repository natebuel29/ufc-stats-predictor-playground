import random
import pandas as pd
import csv
import numpy as np
from logistic_regression_functions import *
from logistic_regression import LogisticRegression
from scipy.optimize import minimize
x_labels = ['rf', 'bf', 'winner', 'rwins', 'bwins', 'rloses', 'bloses', 'rslpm', 'bslpm', 'rstrac', 'bstrac', 'rsapm', 'bsapm', 'rstrd', 'bstrd', 'rtdav',
            'btdav', 'rtdac', 'btdac', 'rtdd', 'btdd', 'rsubav', 'bsubav']
stat_indexes = [4, 5, 13, 14, 15, 16, 17, 18, 19, 20]


def construct_fight_dataframe(df, fighter_stats, shouldRandomize):
    """ 
    Constructs the fight dataframe from using the fights df and fighter stats dict
    Arguments:
        df: the fighter df that is read from fighters.csv
        shouldRandomize: boolean flag to randomize the red/blue corner to balance the classes
    Returns:
        The fight dataframe which includes the fighters,their stats, and the winner
    """
    X = pd.DataFrame(columns=x_labels)
    for row in df.itertuples():
        temp_ar = []
        rwin = row[3]
        chance = random.uniform(0, 1)
        if chance > 0.65 and shouldRandomize:
            rf = row[2]
            bf = row[1]
            rwin = row[4]
            bwin = row[3]
        else:
            rf = row[1]
            bf = row[2]
            rwin = row[3]
            bwin = row[4]

        if rwin != bwin:
            temp_ar.append(rf)
            temp_ar.append(bf)

            winner = 1 if rwin == 1 else 0
            temp_ar.append(winner)

            rf_stats = fighter_stats[rf]
            bf_stats = fighter_stats[bf]

            for index in stat_indexes:
                rstat = rf_stats[index]
                bstat = bf_stats[index]
                temp_ar.append(rstat)
                temp_ar.append(bstat)

            X = pd.concat(
                [pd.DataFrame([temp_ar], columns=x_labels), X], ignore_index=True)

    return X


def main():
    fighter_stats = {}

    with open('data\\fighters.csv', mode='r') as inp:
        reader = csv.reader(inp)
        fighter_stats = {rows[0]: rows[0:] for rows in reader}

    # read from the scraper generated csv files
    fights_df = pd.read_csv('data\\fights.csv')

    # construct a non-randomized dataframe
    fight_df = construct_fight_dataframe(fights_df, fighter_stats, False)
    X = fight_df.loc[:, "rwins":].astype(float).to_numpy()
    X_norm = standardize(X)
    rows, columns = X.shape
    y = fight_df.loc[1:, "winner"].astype(float).to_numpy()
    X = np.concatenate([np.ones((rows, 1)),
                        X], axis=1)
    X_norm = np.concatenate([np.ones((rows, 1)),
                             X_norm], axis=1)
    test = X_norm[0, :]
    print(X)
    print(X_norm)
    X_norm = X_norm[1:, :]
    print(X_norm)
    print(test)
    start_theta = np.zeros(columns+1)
    testLogReg = LogisticRegression(X_norm, y, start_theta)
    iters = 40000
    j_history = testLogReg.gradient_descent(0.01, iters)
    res = minimize(costFunction, start_theta, (X_norm, y),
                   jac=True, method="BFGS", options={'maxiter': 400})

    print(res.x)
    print(testLogReg.theta)
    print("GRADIENT DESCENT AND MINIMIZE PRODUCE THE SAME THETA :)")
    print(predict(res.x, test))


if __name__ == "__main__":
    main()
