# xgboost work through

import xgboost as xgb
import numpy as np
import scipy

# Load data

# To load a libsvm text file
dtrain = xgb.DMatrix('train.svm.txt')
dtest = xgb.DMatrix('test.svm.buffer')

# To load a numpy array
data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
label = np.random.randint(2, size=5)  # binary target
dtrain = xgb.DMatrix(data, label=label)

# To load a scpiy.sparse array
csr = scipy.sparse.csr_matrix((dat, (row, col)))
dtrain = xgb.DMatrix(csr)

# Setting Parameters

# Booster parameters
param = {'bst:max_depth': 2, 'bst:eta': 1,
         'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
evallist = [(dtest, 'eval'), (dtrain, 'train')]

# train
num_round = 10
bst = xgb.train(plst, dtrain, num_round, evallist)

# Early stopping requires at least one set in evals.
# If there’s more than one, it will use the last.

# train(..., evals=evals, early_stopping_rounds=10)

# Prediction
# 7 entities, each contains 10 features
data = np.random.rand(7, 10)
dtest = xgb.DMatrix(data)
ypred = bst.predict(xgmat)

# ypred = bst.predict(xgmat,ntree_limit=bst.best_ntree_limit)

# Plotting
xgb.plot_importance(bst)
