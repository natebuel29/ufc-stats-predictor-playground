import random
import pandas as pd
import csv
import numpy as np
import boto3
import json
import mysql.connector

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
   # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1'
    )
    secretMap = client.get_secret_value(
        SecretId="UfcPredictorRdsSecret-extTBzicS2ON", VersionStage="AWSCURRENT")
    rdsSecret = json.loads(secretMap.get("SecretString"))

    host = rdsSecret.get("host")
    user = rdsSecret.get("username")
    password = rdsSecret.get("password")
    database = rdsSecret.get("dbname")
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
    )

    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM past_matchups")
    fights_df = pd.DataFrame(cursor.fetchall()).loc[:, 1:]

    cursor.execute(f"SELECT * FROM future_matchups WHERE date_='2022-08-13'")

    future_df = pd.DataFrame(cursor.fetchall()).loc[:, 2:]

    X_future = future_df.loc[:, 5:].astype(float).to_numpy()
    X_future = standardize(X_future)

    # lets do a simple 80-20 train-test data split for now but implement cross validation
    # later. I would like to predict the ufc 274 card
    train = fights_df.sample(
        frac=0.8, random_state=250)
    test = fights_df.drop(train.index)

    X_test = test.loc[:, 4:].astype(float).to_numpy()
    X_test = standardize(X_test)

    y_test = test.loc[:, 3].astype(float).to_numpy()
    y = train.loc[:, 3].astype(float).to_numpy()

    X = train.loc[:, 4:].astype(float).to_numpy()
    X = standardize(X)

    return X, y, X_test, y_test, X_future
