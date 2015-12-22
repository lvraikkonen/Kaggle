# ensemble 3 xgboost models
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y = train.QuoteConversion_Flag.values
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test = test.drop(['QuoteNumber'], axis=1)


# treat dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
train = train.drop(['Original_Quote_Date'], axis=1)
test = test.drop(['Original_Quote_Date'], axis=1)

train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['Weekday'] = train['Date'].dt.dayofweek

test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['Weekday'] = test['Date'].dt.dayofweek

train = train.drop(['Date'], axis=1)
test = test.drop(['Date'], axis=1)

# fill NA
train = train.fillna(-1)
test = test.fillna(-1)

# char column encode
for c in train.columns:
    if train[c].dtype == 'object':
        print c
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# train 3 xgboost models
pred_average = True
for k in range(3):
    clf = xgb.XGBClassifier(n_estimators=500,
                            nthread=-1,
                            max_depth=6,
                            learning_rate=0.015,
                            silent=True,
                            subsample=0.85,
                            colsample_bytree=0.65,
                            seed=k * 100 + 22)
    xgb_model = clf.fit(train, y, eval_metric="auc")
    pred = clf.predict_proba(test)[:, 1]
    if type(pred_average) == bool:
        pred_average = pred.copy() * 1.0 / 3.0
    else:
        pred_average += pred * 1.0 / 3.0

sample = pd.read_csv('sample_submission.csv')
sample.QuoteConversion_Flag = pred_average
sample.to_csv('xgb_benchmark.csv', index=False)
