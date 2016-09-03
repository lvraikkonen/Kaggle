import numpy as np
import pandas as pd
import ngram
from nlp_utils import stopwords, english_stemmer, stem_tokens, getTFV
import cPickle
import config
import time

import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from nltk.stem.porter import *
from nltk.metrics import edit_distance
from sklearn.metrics import mean_squared_error, make_scorer

from utility import correct_string

print "Load features..."

start_time = time.time()

df_train = pd.read_csv('input/train.csv', encoding='ISO-8859-1')
df_test = pd.read_csv('input/test.csv', encoding='ISO-8859-1')
df_pro_desc = pd.read_csv('input/product_descriptions.csv')
df_attr = pd.read_csv('input/attributes.csv')
df_brand = df_attr[df_attr.name == "MFG Brand Name"][
    ["product_uid", "value"]].rename(columns={"value": "brand"})

num_train = df_train.shape[0]
y_train = df_train['relevance'].values

# 1. common word feature
df_all = np.loadtxt(config.path_features + 'common_word_feat.txt')
id_test = df_test['id']
X_common_word_feat_train = df_all[:num_train, :]
X_common_word_feat_test = df_all[num_train:, :]

# 2. jaccard coef feature
jaccard_all = np.loadtxt(config.path_features + 'jaccard_dice_dist_feat.txt')
X_jaccard_feat_train = jaccard_all[:num_train, :]
X_jaccard_feat_test = jaccard_all[num_train:, :]

# 3. similiarity feature (word2vec, cosine sim)
X_sim_feat_train = np.loadtxt(
    config.path_features + 'X_similiarity_additional_train.txt')
X_sim_feat_test = np.loadtxt(
    config.path_features + 'X_similiarity_additional_test.txt')

# 4. counts feature
X_train_count = np.loadtxt(config.path_features + 'train_counts.txt')
X_test_count = np.loadtxt(config.path_features + 'test_counts.txt')

# 5-1. extended query count features (top 10)
X_extquery_count_feat_train = np.loadtxt(
    config.path_features + 'train_ext_counts_top10.txt')
X_extquery_count_feat_test = np.loadtxt(
    config.path_features + 'test_ext_counts_top10.txt')

# 6. char similiarity feature
X_char_sim_train = np.loadtxt(config.path_features + 'ssfeas4train.txt')
X_char_sim_test = np.loadtxt(config.path_features + 'ssfeas4test.txt')


# merge all features
X_train = np.hstack((X_common_word_feat_train, X_jaccard_feat_train, X_sim_feat_train,
                     X_train_count, X_extquery_count_feat_train, X_char_sim_train))
X_test = np.hstack((X_common_word_feat_test, X_jaccard_feat_test, X_sim_feat_test,
                    X_test_count, X_extquery_count_feat_test, X_char_sim_test))

print("Load features cost--- %s seconds ---" % (time.time() - start_time))


def fmean_squarded_error(ground_truth, prediction):
    fmean_squared_error_ = mean_squared_error(ground_truth, prediction) ** 0.5
    return fmean_squared_error_

RMSE = make_scorer(fmean_squarded_error, greater_is_better=False)

start_time = time.time()

rfr = RandomForestRegressor()
clf = pipeline.Pipeline([('rfr', rfr)])
param_grid_old = {'rfr__n_estimators': list(
    range(320, 400, 1)), 'rfr__max_depth': list(range(8, 10, 1))}
param_grid = {'rfr__n_estimators': [350],  # list(range(109,110,1)),
              'rfr__max_depth': [8],  # list(range(7,8,1))
              }
model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid,
                                 n_jobs=2, cv=10, verbose=1, scoring=RMSE)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

print("Training model random forest cost--- %s seconds ---" %
      (time.time() - start_time))

y_pred = model.predict(X_test)
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(
    'submission/rf_fe_tuned_20160216.csv', index=False)
