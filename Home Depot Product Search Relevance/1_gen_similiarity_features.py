import config
import pandas as pd
import numpy as np
import scipy.sparse as sp
import re
import cPickle
from nlp_utils import stopwords, english_stemmer, stem_tokens, getTFV
import time
from utility import correct_string


from bs4 import BeautifulSoup
from nltk.stem.porter import *
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import wordnet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances

from gensim.models import Word2Vec
#from tnse import bh_sne


start_time = time.time()

print "Feature Extracting..."
print "Reading Data..."

train_df = pd.read_csv(config.path_train, encoding='ISO-8859-1').fillna('')
test_df = pd.read_csv(config.path_test, encoding='ISO-8859-1').fillna('')
df_pro_desc = pd.read_csv('input/product_descriptions.csv')
train_df = pd.merge(train_df, df_pro_desc, how='left', on='product_uid')
test_df = pd.merge(test_df, df_pro_desc, how='left', on='product_uid')

print("Reading data cost--- %s seconds ---" % (time.time() - start_time))


toker = TreebankWordTokenizer()
lemmer = wordnet.WordNetLemmatizer()


def text_preprocessor(x):
    '''
    Get one string and clean\lemm it
    '''
    tmp = correct_string(x)
    x_cleaned = tmp.replace('/', ' ').replace('-', ' ').replace('"', '')
    tokens = toker.tokenize(x_cleaned)
    return " ".join([lemmer.lemmatize(z) for z in tokens])


# stem description
print "Stemming..."
start_time = time.time()

train_df['desc_stem'] = train_df[
    'product_description'].apply(text_preprocessor)
test_df['desc_stem'] = test_df['product_description'].apply(text_preprocessor)

# stem query
train_df['query_stem'] = train_df['search_term'].apply(text_preprocessor)
test_df['query_stem'] = test_df['search_term'].apply(text_preprocessor)

# stem title
train_df['title_stem'] = train_df['product_title'].apply(text_preprocessor)
test_df['title_stem'] = test_df['product_title'].apply(text_preprocessor)

print("Stermming data cost--- %s seconds ---" % (time.time() - start_time))


# cosine similiarity
print "Calculating cosine similiarity..."


def calc_cosine_dist(text_a, text_b, vect):
    return pairwise_distances(vect.transform([text_a]), vect.transform([text_b]), metric='cosine')[0][0]


def calc_set_intersection(text_a, text_b):
    a = set(text_a.split())
    b = set(text_b.split())
    return len(a.intersection(b)) * 1.0 / len(a)


start_time = time.time()

tfv_orig = getTFV(ngram_range=(1, 2))
tfv_stem = getTFV(ngram_range=(1, 2))
tfv_desc = getTFV(ngram_range=(1, 2))

tfv_orig.fit(list(train_df['search_term'].values) + list(test_df['search_term'].values) +
             list(train_df['product_title'].values) + list(test_df['product_title'].values))
tfv_stem.fit(list(train_df['query_stem'].values) + list(test_df['query_stem'].values) +
             list(train_df['title_stem'].values) + list(test_df['title_stem'].values))
tfv_desc.fit(list(train_df['query_stem'].values) + list(test_df['query_stem'].values) +
             list(train_df['desc_stem'].values) + list(test_df['desc_stem'].values))

print("Vectorizing data cost--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()

# for training set
cosine_orig = []
cosine_stem = []
cosine_desc = []
set_stem = []
for i, row in train_df.iterrows():
    cosine_orig.append(calc_cosine_dist(
        row['search_term'], row['product_title'], tfv_orig))
    cosine_stem.append(calc_cosine_dist(
        row['query_stem'], row['title_stem'], tfv_stem))
    cosine_desc.append(calc_cosine_dist(
        row['query_stem'], row['desc_stem'], tfv_desc))
    set_stem.append(calc_set_intersection(
        row['query_stem'], row['title_stem']))
train_df['cosine_qt_orig'] = cosine_orig
train_df['cosine_qt_stem'] = cosine_stem
train_df['cosine_qd_stem'] = cosine_desc
train_df['set_qt_stem'] = set_stem

print("Calculating cosine similiarity for train set cost--- %s seconds ---" %
      (time.time() - start_time))

start_time = time.time()

# for test set
cosine_orig = []
cosine_stem = []
cosine_desc = []
set_stem = []
for i, row in test_df.iterrows():
    cosine_orig.append(calc_cosine_dist(
        row['search_term'], row['product_title'], tfv_orig))
    cosine_stem.append(calc_cosine_dist(
        row['query_stem'], row['title_stem'], tfv_stem))
    cosine_desc.append(calc_cosine_dist(
        row['query_stem'], row['desc_stem'], tfv_desc))
    set_stem.append(calc_set_intersection(
        row['query_stem'], row['title_stem']))
test_df['cosine_qt_orig'] = cosine_orig
test_df['cosine_qt_stem'] = cosine_stem
test_df['cosine_qd_stem'] = cosine_desc
test_df['set_qt_stem'] = set_stem

print("Calculating cosine similiarity for test set cost--- %s seconds ---" %
      (time.time() - start_time))


# Word2Vec
print "Calculating Word2Vec similiarity and dist..."
start_time = time.time()
embedder = Word2Vec.load_word2vec_format(
    config.path_w2v_pretrained_model, binary=True)


def calc_w2v_sim(row):
    '''
    Calc w2v similarities and diff of centers of query\title
    '''
    a2 = [x for x in row['query_stem'].lower().split() if x in embedder.vocab]
    b2 = [x for x in row['title_stem'].lower().split() if x in embedder.vocab]
    if len(a2) > 0 and len(b2) > 0:
        w2v_sim = embedder.n_similarity(a2, b2)
    else:
        return((-1, -1, np.zeros(300)))

    vectorA = np.zeros(300)
    for w in a2:
        vectorA += embedder[w]
    vectorA /= len(a2)

    vectorB = np.zeros(300)
    for w in b2:
        vectorB += embedder[w]
    vectorB /= len(b2)

    vector_diff = (vectorA - vectorB)

    w2v_vdiff_dist = np.sqrt(np.sum(vector_diff**2))
    return (w2v_sim, w2v_vdiff_dist, vector_diff)


# for train
X_w2v = []
sim_list = []
dist_list = []
for i, row in train_df.iterrows():
    sim, dist, vdiff = calc_w2v_sim(row)
    X_w2v.append(vdiff)
    sim_list.append(sim)
    dist_list.append(dist)
X_w2v_tr = np.array(X_w2v)
train_df['w2v_sim'] = np.array(sim_list)
train_df['w2v_dist'] = np.array(dist_list)

print("Calculating word2vec similiarity for train set cost--- %s seconds ---" %
      (time.time() - start_time))


start_time = time.time()
# for test
X_w2v = []
sim_list = []
dist_list = []
for i, row in test_df.iterrows():
    sim, dist, vdiff = calc_w2v_sim(row)
    X_w2v.append(vdiff)
    sim_list.append(sim)
    dist_list.append(dist)
X_w2v_te = np.array(X_w2v)
test_df['w2v_sim'] = np.array(sim_list)
test_df['w2v_dist'] = np.array(dist_list)

print("Calculating word2vec similiarity for test set cost--- %s seconds ---" %
      (time.time() - start_time))


# # tnse part
# vect = TfidfVectorizer(ngram_range=(1,2), min_df=3)
# X_tf = vect.fit_transform(list(train_df['title_stem'].values) + list(test_df['title_stem'].values))
# svd = TruncatedSVD(n_components=200)
# X_svd = svd.fit_transform(X_tf)
# X_scaled = StandardScaler().fit_transform(X_svd)
# X_tsne = bh_sne(X_scaled)
# train_df['tsne_title_1'] = X_tsne[:len(train_df), 0]
# train_df['tsne_title_2'] = X_tsne[:len(train_df), 1]
# test_df[ 'tsne_title_1'] = X_tsne[len(train_df):, 0]
# test_df[ 'tsne_title_2'] = X_tsne[len(train_df):, 1]

## logging.info('\t [2\3] process title-query')
# vect = TfidfVectorizer(ngram_range=(1,2), min_df=3)
# X_title = vect.fit_transform(list(train_df['title_stem'].values) + list(test_df['title_stem'].values))
# X_query = vect.fit_transform(list(train_df['query_stem'].values) + list(test_df['query_stem'].values))
# X_tf = sp.hstack([X_title, X_query]).tocsr()
# svd = TruncatedSVD(n_components=200)
# X_svd = svd.fit_transform(X_tf)
# X_scaled = StandardScaler().fit_transform(X_svd)
# X_tsne = bh_sne(X_scaled)
# train_df['tsne_qt_1'] = X_tsne[:len(train_df), 0]
# train_df['tsne_qt_2'] = X_tsne[:len(train_df), 1]
# test_df[ 'tsne_qt_1'] = X_tsne[len(train_df):, 0]
# test_df[ 'tsne_qt_2'] = X_tsne[len(train_df):, 1]

## logging.info('\t [3\3] process description')
# vect = TfidfVectorizer(ngram_range=(1,2), min_df=3)
# X_desc = vect.fit_transform(list(train_df['desc_stem'].values) + list(test_df['desc_stem'].values))
# X_tf = X_desc
# svd = TruncatedSVD(n_components=200)
# X_svd = svd.fit_transform(X_tf)
# X_scaled = StandardScaler().fit_transform(X_svd)
# X_tsne = bh_sne(X_scaled)
# train_df['tsne_desc_1'] = X_tsne[:len(train_df), 0]
# train_df['tsne_desc_2'] = X_tsne[:len(train_df), 1]
# test_df[ 'tsne_desc_1'] = X_tsne[len(train_df):, 0]
# test_df[ 'tsne_desc_2'] = X_tsne[len(train_df):, 1]

# # Dump results
train_df.to_pickle(config.path_processed + 'train_df')
test_df.to_pickle(config.path_processed + 'test_df')

# dump generated features
feat_list = [
    u'w2v_sim',
    u'w2v_dist',
    # u'tsne_title_1',
    # u'tsne_title_2',
    # u'tsne_qt_1',
    # u'tsne_qt_2',
    u'cosine_qt_orig',
    u'cosine_qt_stem',
    u'cosine_qd_stem',
    u'set_qt_stem'
]
X_additional_train = train_df[feat_list].as_matrix()
X_additional_test = test_df[feat_list].as_matrix()

np.savetxt(config.path_processed +
           'X_similiarity_additional_train.txt', X_additional_train)
np.savetxt(config.path_processed +
           'X_similiarity_additional_test.txt', X_additional_test)
