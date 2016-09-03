# generate features
import numpy as np
import pandas as pd
import ngram
from nlp_utils import stopwords, english_stemmer, stem_tokens, getTFV
import cPickle

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from nltk.stem.porter import *
from nltk.metrics import edit_distance


# Load dataset
df_train = pd.read_csv('input/train.csv', encoding='ISO-8859-1')
df_test = pd.read_csv('input/test.csv', encoding='ISO-8859-1')
df_pro_desc = pd.read_csv('input/product_descriptions.csv')
num_train = df_train.shape[0]


# clean title (get true product title)
# TBD


# feature engineering
stemmer = PorterStemmer()


def str_stem(str1):
    str1 = str1.lower()
    str1 = str1.replace(" in.", "in.")
    str1 = str1.replace(" inch", "in.")
    str1 = str1.replace("inch", "in.")
    str1 = str1.replace(" in ", "in. ")
    str1 = " ".join([stemmer.stem(z) for z in str1.split(" ")])
    return str1

##########################
## 1. common word count ##
##########################


def str_common_word(str1, str2):
    str1, str2 = str1.lower(), str2.lower()
    words, cnt, words2 = str1.split(), 0, str2.split(),
    for word in words:
        if len(words2) < 10 and len(words) < 4:
            for word2 in words2:
                if edit_distance(word, word2, transpositions=False) <= 1:
                    cnt += 1
        else:
            if str2.find(word) >= 0:
                cnt += 1
    return cnt

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
# left join
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

df_all['search_term'] = df_all['search_term'].map(lambda x: str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stem(x))
df_all['product_description'] = df_all[
    'product_description'].map(lambda x: str_stem(x))

# search term word length
df_all['len_of_query'] = df_all['search_term'].map(
    lambda x: len(x.split())).astype(np.int64)

#
df_all['product_info'] = df_all['search_term'] + "\t" + \
    df_all['product_title'] + "\t" + df_all['product_description']
# common word count
df_all['word_count_in_title'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['word_count_in_description'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))

# df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

# dump df_all file
with open('features/common_word_count_feat.pkl', 'wb') as f:
    cPickle.dump(df_all, f)
    del df_all


#################################
## 2. Jaccard coef & Dice dist ##
#################################
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


# Load features generated last step
with open('features/common_word_count_feat.pkl', 'rb') as f:
    df_all = cPickle.load(f)

print "Generate distince features..."
extract_basic_distance_feat(df_all)

# dump df_all file
with open('features/jaccard_dice_dist_feat.pkl', 'wb') as f:
    cPickle.dump(df_all, f)
    del df_all


#################################
## 3. Co-occurrence TF-IDF     ##
#################################
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


# load data from last step
with open('features/jaccard_dice_dist_feat.pkl', 'rb') as f:
    df_all = cPickle.load(f)


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
extract_cooccurrence_feature(df_all)
# extract_cooccurrence_feature(df_train)
# extract_cooccurrence_feature(df_test)

print "For training and testing..."

for feat_name, column_name in zip(feat_names, column_names):
    print "Generate %s feature" % feat_name
    tfv = getTFV(ngram_range=ngram_range)
    X_tfidf_all = tfv.fit_transform(df_all[column_name])
    #X_tfidf_train = tfv.fit_transform(df_train[column_name])
    #X_tfidf_test = tfv.transform(df_test[column_name])

    # SVD
    svd = TruncatedSVD(n_components=svd_n_component, n_iter=15)
    X_svd_all = svd.fit_transform(X_tfidf_all)
    #X_svd_train = svd.fit_transform(X_tfidf_train)
    #X_svd_test = svd.transform(X_tfidf_test)
