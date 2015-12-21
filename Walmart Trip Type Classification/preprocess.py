# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd


def walmart_prepare(df):

    def convert_weekday(day):
        '''Function to convert weekday char strings
        to numeric'''
        if day == 'Sunday':
            return 0
        elif day == 'Monday':
            return 1
        elif day == 'Tuesday':
            return 2
        elif day == 'Wednesday':
            return 3
        elif day == 'Thursday':
            return 4
        elif day == 'Friday':
            return 5
        else:
            return 6

    # Drop UPC (there are too many to deal with), fill NAs, and convert
    # weekdays to numeric.
    df.drop(['Upc'], inplace=True, axis=1)
    df.DepartmentDescription = df.DepartmentDescription.fillna('None')
    df.FinelineNumber = df.FinelineNumber.fillna(-1)
    df.Weekday = df.Weekday.apply(convert_weekday)
    # Add a column for returns
    df['Returns'] = pd.Series([abs(num) if num < 0
                               else 0 for num in df.ScanCount], index=df.index)
    df.ScanCount = df.ScanCount.apply(lambda x: 0 if x < 0 else x)
    # Rename ScanCount to Purchases
    df.rename(columns={'ScanCount': 'Purchases'}, inplace=True)

    # Create dummy variables for dept description and fineline number.
    # (Dropping the first column of each to avoid multicollinearity.)
    temp1 = pd.get_dummies(df.DepartmentDescription).astype(int, copy=False)
    temp2 = pd.get_dummies(df.FinelineNumber).astype(int, copy=False)
    temp1.drop(temp1.columns[0], inplace=True, axis=1)
    temp2.drop(temp2.columns[0], inplace=True, axis=1)

    # Now we don't need these columns anymore
    df.drop(['DepartmentDescription', 'FinelineNumber'], inplace=True, axis=1)

    # Concatenate df with the dummy dataframes after converting df to int
    df.astype(int, copy=False)
    df = pd.concat([df, temp1, temp2], axis=1)

    del temp1, temp2

    # for train.csv
    if 'TripType' in df.columns:
        return df.groupby(['TripType', 'VisitNumber', 'Weekday']).aggregate(np.sum).reset_index()

    # for test.csv
    else:
        return df.groupby(['VisitNumber', 'Weekday']).aggregate(np.sum).reset_index()


if __name__ == "__main__":

    import xgboost as xgb
    import random

    from sklearn.cross_validation import train_test_split

    # This all has to be done in parts because my computer doesn't have
    # enough RAM.
    train = pd.read_csv('data/train.csv')

    # XGB needs target classes to be in [0, num_classes)
    ref = {type: n for (type, n) in zip(
        np.sort(train.TripType.unique()), range(38))}
    train.TripType = train.TripType.apply(lambda x: ref[x])
    test = pd.read_csv('data/test.csv')

    print "Preparing Walmart training set with 80/20 split for CV!"
    random.seed(a=0)
    rows = random.sample(train.index, 300000)
    train_prepare1 = walmart_prepare(train.ix[rows])
    train_prepare2 = walmart_prepare(train.drop(rows))

    X1 = train_prepare1.drop(['TripType'], axis=1)
    y1 = train_prepare1.TripType
    X2 = train_prepare2.drop(['TripType'], axis=1)
    y2 = train_prepare2.TripType
    del train_prepare1, train_prepare2

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2, random_state=0)
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=0)
    del X1, y1, X2, y2

    print "Saving training DMatrices to XGBoost binary buffer files!"
    dtrain1 = xgb.DMatrix(np.array(X1_train), label=np.array(y1_train))
    dtrain2 = xgb.DMatrix(np.array(X2_train), label=np.array(y2_train))
    dtrain1.save_binary('data/dtrain1.buffer')
    dtrain2.save_binary('data/dtrain2.buffer')
    del dtrain1, dtrain2, X1_train, X2_train, y1_train, y2_train

    print "Saving CV test DMatrices to XGBoost binary buffer files!"
    dtestCV1 = xgb.DMatrix(np.array(X1_test), label=np.array(y1_test))
    dtestCV2 = xgb.DMatrix(np.array(X2_test), label=np.array(y2_test))
    dtestCV1.save_binary('data/dtestCV1.buffer')
    dtestCV2.save_binary('data/dtestCV2.buffer')
    del dtestCV1, dtestCV2, X1_test, X2_test, y1_test, y2_test

    print "Preparing Walmart testing set!"
    test_prepare1 = walmart_prepare(test.ix[rows])
    test_prepare2 = walmart_prepare(test.drop(rows))

    print "Saving test sets to an XGBoost binary buffer file!"
    dtest1 = xgb.DMatrix(np.array(test_prepare1))
    dtest2 = xgb.DMatrix(np.array(test_prepare2))
    dtest1.save_binary('data/test1.buffer')
    dtest2.save_binary('data/test2.buffer')
