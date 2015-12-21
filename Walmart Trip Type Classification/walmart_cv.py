# -*- coding: utf-8 -*-

import numpy as np
import xgboost as xgb

from sklearn.metrics import log_loss

from hyperopt import fmin, tpe, hp, STATUS_OK


# These functions define the metric which we are trying to
# optimize.
def objective1(params):
    print "Training model1 with parameters: "
    print params
    watchlist1 = [(dtrain1, 'train'), (dtestCV1, 'eval')]
    model = xgb.train(params=params,
                      dtrain=dtrain1,
                      num_boost_round=1000,
                      early_stopping_rounds=10,
                      evals=watchlist1)
    score = log_loss(dtestCV1.get_label(), model.predict(dtestCV1))
    print "\tScore {0}\n\n".format(score)
    return {'loss': score, 'status': STATUS_OK}


def objective2(params):
    print "Training model2 with parameters: "
    print params
    watchlist2 = [(dtrain2, 'train'), (dtestCV2, 'eval')]
    model = xgb.train(params=params,
                      dtrain=dtrain1,
                      num_boost_round=1000,
                      early_stopping_rounds=10,
                      evals=watchlist2)
    score = log_loss(dtestCV2.get_label(), model.predict(dtestCV2))
    print "\tScore {0}\n\n".format(score)
    return {'loss': score, 'status': STATUS_OK}


# Load data from buffer files
dtrain1 = xgb.DMatrix('data/dtrain1.buffer')
dtestCV1 = xgb.DMatrix('data/dtestCV1.buffer')
dtrain2 = xgb.DMatrix('data/dtrain2.buffer')
dtestCV2 = xgb.DMatrix('data/dtestCV2.buffer')


# Define the hyperparameter space
space = {'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
         'max_depth': hp.quniform('max_depth', 1, 15, 1),
         'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
         'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
         'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
         'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
         'num_class': 38,
         'eval_metric': 'mlogloss',
         'objective': 'multi:softprob'}


# Evaluate the function fmin over the hyperparameter space, and
# print the best hyperparameters.
best1 = fmin(objective1, space=space, algo=tpe.suggest, max_evals=250)
print "Optimal parameters for dtrain1 are: ", best1
#

best2 = fmin(objective2, space=space, algo=tpe.suggest, max_evals=250)
print "Optimal parameters for dtrain2 are: ", best2
#
