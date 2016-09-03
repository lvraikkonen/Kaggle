import numpy as np
import pandas as pd
import ngram
from nlp_utils import stopwords, english_stemmer, stem_tokens, getTFV
import cPickle
import config

import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from nltk.stem.porter import *
from nltk.metrics import edit_distance


# read dataset
df_train = pd.read_csv('input/train.csv', encoding='ISO-8859-1')
df_test = pd.read_csv('input/test.csv', encoding='ISO-8859-1')
df_pro_desc = pd.read_csv('input/product_descriptions.csv')
num_train = df_train.shape[0]


def cooccurrence_terms(lst1, lst2, join_str):
    terms = [""] * len(lst1) * len(lst2)
    cnt = 0
    for item1 in lst1:
        for item2 in lst2:
            terms[cnt] = item1 + join_str + item2
            cnt += 1
    res = " ".join(terms)
    return res


def extract_cooccurrence_feature(df):
    #    ## unigram
    #    print "generate unigram"
    #    df["query_unigram"] = list(df.apply(lambda x: preprocess_data(x["search_term"]), axis=1))
    #    df["title_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_title"]), axis=1))
    #    df["description_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_description"]), axis=1))
    #    ## bigram
    #    print "generate bigram"
    #    join_str = "_"
    #    df["query_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["query_unigram"], join_str), axis=1))
    #    df["title_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["title_unigram"], join_str), axis=1))
    #    df["description_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["description_unigram"], join_str), axis=1))
    #    # ## trigram
    #    # join_str = "_"
    #    # df["query_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["query_unigram"], join_str), axis=1))
    #    # df["title_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["title_unigram"], join_str), axis=1))
    #    # df["description_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["description_unigram"], join_str), axis=1))

    # cooccurrence terms
    join_str = "X"
    # query unigram
    df["query_unigram_title_unigram"] = list(df.apply(lambda x: cooccurrence_terms(
        x["term_unigram"], x["title_unigram"], join_str), axis=1))
    df["query_unigram_title_bigram"] = list(df.apply(lambda x: cooccurrence_terms(
        x["term_unigram"], x["title_bigram"], join_str), axis=1))
    df["query_unigram_description_unigram"] = list(df.apply(lambda x: cooccurrence_terms(
        x["term_unigram"], x["description_unigram"], join_str), axis=1))
    df["query_unigram_description_bigram"] = list(df.apply(lambda x: cooccurrence_terms(
        x["term_unigram"], x["description_bigram"], join_str), axis=1))
    # query bigram
    df["query_bigram_title_unigram"] = list(df.apply(lambda x: cooccurrence_terms(
        x["term_bigram"], x["title_unigram"], join_str), axis=1))
    df["query_bigram_title_bigram"] = list(df.apply(lambda x: cooccurrence_terms(
        x["term_bigram"], x["title_bigram"], join_str), axis=1))
    df["query_bigram_description_unigram"] = list(df.apply(lambda x: cooccurrence_terms(
        x["term_bigram"], x["description_unigram"], join_str), axis=1))
    df["query_bigram_description_bigram"] = list(df.apply(lambda x: cooccurrence_terms(
        x["term_bigram"], x["description_bigram"], join_str), axis=1))


# cooccurrence terms column names
column_names = [
    "query_unigram_title_unigram",
    "query_unigram_title_bigram",
    "query_unigram_description_unigram",
    "query_unigram_description_bigram",
    "query_bigram_title_unigram",
    "query_bigram_title_bigram",
    "query_bigram_description_unigram",
    "query_bigram_description_bigram"
]
# feature names
feat_names = [name + "_tfidf" for name in column_names]
ngram_range = (1, 3)
svd_n_component = 100

# Generate co-occurrence tfidf feature
extract_cooccurrence_feature(df_train)
extract_cooccurrence_feature(df_test)

print "For training and testing..."

for feat_name, column_name in zip(feat_names, column_names):
    print "Generate %s feature" % feat_name
    tfv = getTFV(ngram_range=ngram_range)
    X_tfidf_train = tfv.fit_transform(df_train[column_name])
    X_tfidf_test = tfv.transform(df_test[column_name])
    with open("%s/train_%s_feat.pkl" % (config.path_features, feat_name), "wb") as f:
        cPickle.dump(X_tfidf_train, f, -1)
    with open("%s/test_%s_feat.pkl" % (config.path_features, feat_name), "wb") as f:
        cPickle.dump(X_tfidf_test, f, -1)

    # SVD
    svd = TruncatedSVD(n_components=svd_n_component, n_iter=15)
    X_svd_train = svd.fit_transform(X_tfidf_train)
    X_svd_test = svd.transform(X_tfidf_test)
    with open("%s/train_%s_individual_svd%d_feat.pkl" % (config.path_features, feat_name, svd_n_component), "wb") as f:
        cPickle.dump(X_svd_train, f, -1)
    with open("%s/test_%s_individual_svd%d_feat.pkl" % (config.path_features, feat_name, svd_n_component), "wb") as f:
        cPickle.dump(X_svd_test, f, -1)
