from distutils.util import rfc822_escape
import random
import sys
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

    future_df = pd.read_csv('data\\future_fights.csv')
    # only grab fights for May 7th card
    future_df = construct_fight_dataframe(
        future_df.loc[future_df["date"] == "May 07, 2022"], fighter_stats, False)

    future_X = future_df.loc[:, "rwins":].astype(float).to_numpy()
    future_X = standardize(future_X)
    future_X = np.concatenate([np.ones((future_X.shape[0], 1)),
                               future_X], axis=1)
    # construct a non-randomized dataframe
    fights_df = construct_fight_dataframe(fights_df, fighter_stats, True)

    # lets do a simple 80-20 train-test data split for now but implement cross validation
    # later. I would like to predict the ufc 274 card
    train = fights_df.sample(
        frac=0.8, random_state=250)
    test = fights_df.drop(train.index)

    test_X = test.loc[:, "rwins":].astype(float).to_numpy()
    test_X = standardize(test_X)
    test_X = np.concatenate([np.ones((test_X.shape[0], 1)),
                             test_X], axis=1)
    test_y = test.loc[:, "winner"].astype(float).to_numpy()

    X = train.loc[:, "rwins":].astype(float).to_numpy()
    X_norm = standardize(X)
    rows, columns = X.shape
    y = train.loc[:, "winner"].astype(float).to_numpy()
    X = np.concatenate([np.ones((rows, 1)),
                        X], axis=1)
    X_norm = np.concatenate([np.ones((rows, 1)),
                             X_norm], axis=1)
    start_theta = np.zeros(columns+1)

    # use the scipy.optimize.minimize function to train our parameters
    # compared the generated weights and they matched my gradient descent function (:
    res = minimize(costFunction, start_theta, (X_norm, y),
                   jac=True, method="BFGS", options={'maxiter': 400})

    correct = 0
    total = test_y.size
    guesses = []
    print(total)
    for i in range(0, total):
        x = test_X[i]
        result = predict(res.x, x)
        guesses.append(result)
        if result == test_y[i]:
            correct += 1

    print(f"test results are {correct/total}")

    for i in range(0, future_X.shape[0]):
        x = future_X[i]
        fight_details = future_df.loc[i, :]
        rf = fight_details['rf']
        bf = fight_details['bf']
        result = predict(res.x, x)
        winner = rf if result == 1 else bf

        print(f"The predicted winner of {rf} vs {bf} is: {winner}")


if __name__ == "__main__":
    main()
