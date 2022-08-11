import random
import pandas as pd
import csv
import numpy as np

from logistic_regression.logistic_regression_functions import *

fdf_labels = ['rf', 'bf', 'winner', 'rwins', 'bwins', 'rloses', 'bloses', 'rslpm', 'bslpm', 'rstrac', 'bstrac', 'rsapm', 'bsapm', 'rstrd', 'bstrd', 'rtdav',
              'btdav', 'rtdac', 'btdac', 'rtdd', 'btdd', 'rsubav', 'bsubav']
x_labels = ['rwins', 'bwins', 'rloses', 'bloses', 'rslpm', 'bslpm', 'rstrac', 'bstrac', 'rsapm', 'bsapm', 'rstrd', 'bstrd', 'rtdav',
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
    X = pd.DataFrame(columns=fdf_labels)
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
                [pd.DataFrame([temp_ar], columns=fdf_labels), X], ignore_index=True)

    return X


def construct_data():
    # lets figure out how to compute confidence scores
    fighter_stats = {}

    with open('data\\fighters.csv', mode='r') as inp:
        reader = csv.reader(inp)
        fighter_stats = {rows[0]: rows[0:] for rows in reader}

    # read from the scraper generated csv files
    fights_df = pd.read_csv('data\\fights.csv')

    future_df = pd.read_csv('data\\future_fights.csv')
    # only grab fights for May 7th card
    future_df = construct_fight_dataframe(
        future_df.loc[future_df["date"] == "May 14, 2022"], fighter_stats, False)

    X_future = future_df.loc[:, "rwins":].astype(float).to_numpy()
    X_future = standardize(X_future)

    # construct a non-randomized dataframe
    fights_df = construct_fight_dataframe(fights_df, fighter_stats, True)

    # lets do a simple 80-20 train-test data split for now but implement cross validation
    # later. I would like to predict the ufc 274 card
    train = fights_df.sample(
        frac=0.8, random_state=250)
    test = fights_df.drop(train.index)

    X_test = test.loc[:, "rwins":].astype(float).to_numpy()
    X_test = standardize(X_test)

    y_test = test.loc[:, "winner"].astype(float).to_numpy()
    y = train.loc[:, "winner"].astype(float).to_numpy()

    X = train.loc[:, "rwins":].astype(float).to_numpy()
    X = standardize(X)

    return X, y, X_test, y_test, X_future
