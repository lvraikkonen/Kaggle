import config
import pandas as pd
import numpy as np
import scipy.sparse as sp
import re
import cPickle as pickle
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
    tmp = unicode(x)
    tmp = tmp.lower().replace('blu-ray', 'bluray').replace('wi-fi', 'wifi')
    tmp = correct_string(tmp)
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


# dump generated features
feat_list = [
    u'w2v_sim',
    u'w2v_dist'
    # u'tsne_title_1',
    # u'tsne_title_2',
    # u'tsne_qt_1',
    # u'tsne_qt_2',
]
X_word2vec_sim_train = train_df[feat_list].as_matrix()
X_word2vec_sim_test = test_df[feat_list].as_matrix()

np.savetxt(config.path_processed +
           'X_word2vec_sim_train.txt', X_word2vec_sim_train)
np.savetxt(config.path_processed +
           'X_word2vec_sim_test.txt', X_word2vec_sim_test)
