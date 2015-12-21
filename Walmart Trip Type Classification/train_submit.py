# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
import random

import xgboost as xgb

from preprocess import walmart_prepare


train = pd.read_csv('data/train.csv')
ref = {type: n for (type, n) in zip(
    np.sort(train.TripType.unique()), range(38))}
train.TripType = train.TripType.apply(lambda x: ref[x])

print "Preparing Walmart training set for XGBoost!"
random.seed(a=0)
rows = random.sample(train.index, 300000)
train_prepare1 = walmart_prepare(train.ix[rows])
train_prepare2 = walmart_prepare(train.drop(rows))

X1 = train_prepare1.drop(['TripType'], axis=1)
y1 = train_prepare1.TripType
X2 = train_prepare2.drop(['TripType'], axis=1)
y2 = train_prepare2.TripType

dtrain1 = xgb.DMatrix(np.array(X1), label=np.array(y1))
dtrain2 = xgb.DMatrix(np.array(X2), label=np.array(y2))
del train_prepare1, train_prepare2, X1, y1, X2, y2


# Set hyperparameters.
# default values for xgboost
param = {'colsample_bytree': 0.7,
         'min_child_weight': 2,
         'subsample': 1,
         'eta': 0.35,
         'max_depth': 14,
         'gamma': 0.7,
         'objective': 'multi:softprob',
         'eval_metric': 'mlogloss',
         'num_class': 38,
         'early_stopping_rounds': 10}
num_round = 1500

print "Training XGBoost!"
bst1 = xgb.train(param, dtrain1, num_round)
bst2 = xgb.train(param, dtrain2, num_round)


print "Importing Walmart test sets!"
dtest1 = xgb.DMatrix('data/test1.buffer')
dtest2 = xgb.DMatrix('data/test2.buffer')


print "Making predictions!"
pred11 = bst1.predict(dtest1)
pred21 = bst2.predict(dtest1)

pred12 = bst1.predict(dtest2)
pred22 = bst2.predict(dtest2)


print "Writing to csv!"
test1 = pd.read_csv('data/test.csv').ix[rows]
test2 = pd.read_csv('data/test.csv').drop(rows)

visit_num1 = test1.VisitNumber.unique()
visit_num2 = test2.VisitNumber.unique()

col_names = ['VisitNumber', 'TripType_3', 'TripType_4', 'TripType_5', 'TripType_6', 'TripType_7',
             'TripType_8', 'TripType_9', 'TripType_12', 'TripType_14', 'TripType_15', 'TripType_18',
             'TripType_19', 'TripType_20', 'TripType_21', 'TripType_22', 'TripType_23', 'TripType_24',
             'TripType_25', 'TripType_26', 'TripType_27', 'TripType_28', 'TripType_29', 'TripType_30',
             'TripType_31', 'TripType_32', 'TripType_33', 'TripType_34', 'TripType_35', 'TripType_36',
             'TripType_37', 'TripType_38', 'TripType_39', 'TripType_40', 'TripType_41', 'TripType_42',
             'TripType_43', 'TripType_44', 'TripType_999']

submission11 = pd.DataFrame(pred11, index=visit_num1).reset_index()
submission11.columns = col_names
submission21 = pd.DataFrame(pred21, index=visit_num1).reset_index()
submission21.columns = col_names
submission12 = pd.DataFrame(pred12, index=visit_num2).reset_index()
submission12.columns = col_names
submission22 = pd.DataFrame(pred22, index=visit_num2).reset_index()
submission22.columns = col_names

submission = pd.concat([submission11, submission21,
                        submission12, submission22]).groupby('VisitNumber').mean()

submission.to_csv('data/submission_default.csv')
