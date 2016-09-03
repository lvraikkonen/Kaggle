import numpy as np
import pandas as pd
import ngram
from nlp_utils import stopwords, english_stemmer, stem_tokens, getTFV
import cPickle
from utility import correct_string

import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from nltk.stem.porter import *
from nltk.metrics import edit_distance
import time


# read dataset
df_train = pd.read_csv('input/train.csv', encoding='ISO-8859-1')
df_test = pd.read_csv('input/test.csv', encoding='ISO-8859-1')
df_pro_desc = pd.read_csv('input/product_descriptions.csv')
num_train = df_train.shape[0]


stemmer = PorterStemmer()


def str_stem(s):
    if isinstance(s, str):
        s = s.lower()
        s = correct_string(s)
        s = " ".join([stemmer.stem(re.sub('[^A-Za-z0-9-./]', ' ', word))
                      for word in s.lower().split(" ")])
        return s
    else:
        return "null"

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all['search_term'] = df_all['search_term'].map(lambda x: str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stem(x))
df_all['product_description'] = df_all[
    'product_description'].map(lambda x: str_stem(x))


def try_divided(x, y, val=0.0):
    if y != 0.0:
        val = float(x) / y
    return val

# Jaccard coefficient between search_term and title & search_term and
# description


def jaccardCoef(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = try_divided(intersect, union)
    return coef


def diceDist(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = try_divided(2 * intersect, union)
    return d


def compute_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = jaccardCoef(A, B)
    elif dist == "dice_dist":
        d = diceDist(A, B)
    return d


def pairwise_jaccard_coef(A, B):
    coef = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            coef[i, j] = jaccardCoef(A[i], B[j])
    return coef


def pairwise_jaccard_coef(A, B):
    d = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            d[i, j] = diceDist(A[i], B[j])
    return d

token_pattern = r"(?u)\b\w\w+\b"


def preprocess_data(line, token_pattern=token_pattern, encode_digit=False):
    token_pattern = re.compile(token_pattern, flags=re.UNICODE | re.LOCALE)
    # tokenize
    tokens = [x.lower() for x in token_pattern.findall(line)]
    # stem
    tokens_stemmed = stem_tokens(tokens, english_stemmer)

    return tokens_stemmed


def extract_basic_distance_feat(df):
    # unigram
    print "generate unigram"
    df["term_unigram"] = list(
        df.apply(lambda x: preprocess_data(x["search_term"]), axis=1))
    df["title_unigram"] = list(
        df.apply(lambda x: preprocess_data(x["product_title"]), axis=1))
    df["description_unigram"] = list(
        df.apply(lambda x: preprocess_data(x["product_description"]), axis=1))
    # bigram
    print "generate bigram"
    join_str = "_"
    df["term_bigram"] = list(
        df.apply(lambda x: ngram.getBigram(x["term_unigram"], join_str), axis=1))
    df["title_bigram"] = list(
        df.apply(lambda x: ngram.getBigram(x["title_unigram"], join_str), axis=1))
    df["description_bigram"] = list(df.apply(
        lambda x: ngram.getBigram(x["description_unigram"], join_str), axis=1))
    # trigram
    print "generate trigram"
    join_str = "_"
    df["term_trigram"] = list(
        df.apply(lambda x: ngram.getTrigram(x["term_unigram"], join_str), axis=1))
    df["title_trigram"] = list(
        df.apply(lambda x: ngram.getTrigram(x["title_unigram"], join_str), axis=1))
    df["description_trigram"] = list(df.apply(
        lambda x: ngram.getTrigram(x["description_unigram"], join_str), axis=1))

    # jaccard coef/dice dist of n-gram
    print "generate jaccard coef and dice dist for n-gram"
    dists = ["jaccard_coef", "dice_dist"]
    grams = ["unigram", "bigram", "trigram"]
    feat_names = ["term", "title", "description"]
    for dist in dists:
        for gram in grams:
            for i in range(len(feat_names) - 1):
                for j in range(i + 1, len(feat_names)):
                    target_name = feat_names[i]
                    obs_name = feat_names[j]
                    df["%s_of_%s_between_%s_%s" % (dist, gram, target_name, obs_name)] = \
                        list(df.apply(lambda x: compute_dist(
                            x[target_name + "_" + gram], x[obs_name + "_" + gram], dist), axis=1))


print "Generate jaccard distince features..."
start_time = time.time()
extract_basic_distance_feat(df_all)
print("Calculating jaccard coef cost--- %s seconds ---" %
      (time.time() - start_time))
print df_all.columns

# dump generated features
# np.savetxt('features/jaccard_dice_dist_feat.txt', df_all)
with open('features/jaccard_dice_dist_feat.pkl', 'wb') as f:
    cPickle.dump(df_all, f)
