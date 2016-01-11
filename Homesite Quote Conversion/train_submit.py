import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import math


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

rng = np.random.RandomState(1024)


def feature_engineering(train_df, test_df):
    # process Date
    train_df['Date'] = pd.to_datetime(
        pd.Series(train_df['Original_Quote_Date']))
    train_df = train_df.drop('Original_Quote_Date', axis=1)

    test_df['Date'] = pd.to_datetime(pd.Series(test_df['Original_Quote_Date']))
    test_df = test_df.drop('Original_Quote_Date', axis=1)

    train_df['Year'] = train_df['Date'].apply(lambda x: int(str(x)[:4]))
    train_df['Month'] = train_df['Date'].apply(lambda x: int(str(x)[5:7]))
    train_df['weekday'] = train_df['Date'].dt.dayofweek

    test_df['Year'] = test_df['Date'].apply(lambda x: int(str(x)[:4]))
    test_df['Month'] = test_df['Date'].apply(lambda x: int(str(x)[5:7]))
    test_df['weekday'] = test_df['Date'].dt.dayofweek

    train_df = train_df.drop('Date', axis=1)
    test_df = test_df.drop('Date', axis=1)

    # fill NA
    train_df = train_df.fillna(-999)
    test_df = test_df.fillna(-999)

    # count 0s and NAs
    def count_num(df):
        df['Below0'] = np.sum(df < 0.0, axis=1)
        cols = [col for col in df.columns if col != 'QuoteConversion_Flag']
        df['Equal0'] = np.sum(df[cols] == 0.0, axis=1)
        return df

    train_df = count_num(train_df)
    test_df = count_num(test_df)

    # encode non-numeric
    for col in train_df.columns:
        # print col, train_df[col].dtype
        if train_df[col].dtype == 'object':
            encoder = preprocessing.LabelEncoder()
            encoder.fit(list(train_df[col].values) + list(test_df[col].values))
            train_df[col] = encoder.transform(list(train_df[col].values))
            test_df[col] = encoder.transform(list(test_df[col].values))

    return train_df, test_df

train_df, test_df = feature_engineering(train_df, test_df)

train_features, val_features, train_target, val_target = train_test_split(train_df.drop(
    ['QuoteNumber', 'QuoteConversion_Flag'], axis=1), train_df['QuoteConversion_Flag'].values, test_size=0.1, random_state=88)

# cross validate
print "Begin Cross Validation:"

xbg_params = {
    'max_depth': 10,
    'eta': 0.01,
    'objective': "binary:logistic",
    'booster': "gbtree",
    'eval_metric': 'auc',
    'nthread': 4,
    'subsample': 0.83,
    'colsample_bytree': 0.77,
    'early_stopping_rounds': 10
}

cv_nround = 6000
cv_nfold = 5

X_dev = train_df.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
y_dev = train_df['QuoteConversion_Flag'].values
test_index = test_df['QuoteNumber'].values
X_test = test_df.drop('QuoteNumber', axis=1)

dtrain = xgb.DMatrix(X_dev, label=y_dev)

bst_cv = xgb.cv(xbg_params, dtrain, nfold=cv_nfold, num_boost_round=cv_nround, metrics=['auc'],
                early_stopping_rounds=10, show_progress=True)


# find best hyper-params for xgboost
def find_best_params(X_train, y_train):
    print("Train a XGBoost model")

    train_features, train_target = X_train, y_train

    xgb_model = xgb.XGBClassifier()

    # when in doubt, use xgboost
    parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
                  'objective': ['binary:logistic'],
                  'learning_rate': [0.05],  # so called `eta` value
                  'max_depth': [8],
                  'min_child_weight': [11],
                  'silent': [1],
                  'subsample': [0.83],
                  'colsample_bytree': [0.77],
                  'n_estimators': [600, 900, 1200],  # number of trees
                  'seed': [1024]}

    # evaluate with roc_auc_truncated
    def _score_func(estimator, X, y):
        pred_probs = estimator.predict_proba(X)[:, 1]
        return roc_auc_truncated(y, pred_probs)

    # should evaluate by train_eval instead of the full dataset
    clf = GridSearchCV(xgb_model, parameters, n_jobs=4,
                       cv=StratifiedKFold(
                           train_target, n_folds=3, shuffle=True),
                       # scoring=_score_func,
                       verbose=2, refit=True, scoring='roc_auc')

    clf.fit(train_features, train_target)

    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    return best_parameters

dtrain = xgb.DMatrix(train_features, label=train_target)
dval = xgb.DMatrix(val_features, label=val_target)

params = {
    'max_depth': 6,
    'eta': 0.01,
    'objective': "binary:logistic",
    'booster': "gbtree",
    'eval_metric': 'auc',
    'nthread': 2,
    'subsample': 0.83,
    'colsample_bytree': 0.77,
    'early_stopping_rounds': 10
}

num_round = 6000
watchlist = [(dtrain, 'train'), (dval, 'validation')]

clf = xgb.train(params, dtrain, num_round, watchlist)

test_features = test_df.drop('QuoteNumber', axis=1)
test_index = test_df['QuoteNumber'].values
dtest = xgb.DMatrix(test_features)
preds = clf.predict(dtest)

submit_df = pd.DataFrame(
    {"QuoteNumber": test_index, "QuoteConversion_Flag": preds})
submit_df.to_csv('homesite_20151231_6000Rounds_0_01eta.csv', index=False)
