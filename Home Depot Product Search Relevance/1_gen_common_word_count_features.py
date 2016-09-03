# -*- coding: utf-8 -*-
# import codecs
# import numpy as np
# import pandas as pd
# import ngram
# from nlp_utils import stopwords, english_stemmer, stem_tokens, getTFV
# import cPickle
# from utility import correct_string
import config

# import re
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn import decomposition, pipeline, metrics, grid_search
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import StandardScaler
# import xgboost as xgb
# from nltk.stem.porter import *
# from nltk.metrics import edit_distance
# import time


# read dataset
# df_train = pd.read_csv('input/train.csv', encoding='ISO-8859-1')
# df_test = pd.read_csv('input/test.csv', encoding='ISO-8859-1')
# df_pro_desc = pd.read_csv('input/product_descriptions.csv')
# df_attr = pd.read_csv('input/attributes.csv')
# df_brand = df_attr[df_attr.name == "MFG Brand Name"][
#     ["product_uid", "value"]].rename(columns={"value": "brand"})

# num_train = df_train.shape[0]

# stemmer = PorterStemmer()


# def str_stem(s):
#     if isinstance(s, str):
#         s = s.lower()
#         s = correct_string(s)
#         s = " ".join([stemmer.stem(re.sub('[^A-Za-z0-9-./]', ' ', word))
#                       for word in s.split(" ")])
#         return s
#     else:
#         return "null"


# def str_common_word(str1, str2):
#     str1, str2 = str1.lower(), str2.lower()
#     words, cnt = str1.split(), 0
#     for word in words:
#         if str2.find(word) >= 0:
#             cnt += 1
#     return cnt


# def str_whole_word(str1, str2, i_):
#     str1, str2 = str1.lower().strip(), str2.lower().strip()
#     cnt = 0
#     # if len(str1)>0 and len(str2)>0:
#     #    cnt = len(re.findall(str1,str2))
#     while i_ < len(str2):
#         i_ = str2.find(str1, i_)
#         if i_ == -1:
#             return cnt
#         else:
#             cnt += 1
#             i_ += len(str1)
#     return cnt

# start_time = time.time()
# df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
# df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
# df_all = pd.merge(df_all, df_brand, how='left', on="product_uid")

# df_all['search_term'] = df_all['search_term'].map(lambda x: str_stem(x))
# df_all['product_title'] = df_all['product_title'].map(lambda x: str_stem(x))
# df_all['product_description'] = df_all[
#     'product_description'].map(lambda x: str_stem(x))

# df_all['brand'] = df_all['brand'].map(lambda x: str_stem(x))
# df_all['len_of_query'] = df_all['search_term'].map(
#     lambda x: len(x.split())).astype(np.int64)
# df_all['len_of_title'] = df_all['product_title'].map(
#     lambda x: len(x.split())).astype(np.int64)
# df_all['len_of_description'] = df_all['product_description'].map(
#     lambda x: len(x.split())).astype(np.int64)
# df_all['len_of_brand'] = df_all['brand'].map(
#     lambda x: len(x.split())).astype(np.int64)
# df_all['product_info'] = df_all['search_term'] + "\t" + \
#     df_all['product_title'] + "\t" + df_all['product_description']
# df_all['query_in_title'] = df_all['product_info'].map(
#     lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[1], 0))
# df_all['query_in_description'] = df_all['product_info'].map(
#     lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[2], 0))
# df_all['word_in_title'] = df_all['product_info'].map(
#     lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
# df_all['word_in_description'] = df_all['product_info'].map(
#     lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
# df_all['ratio_title'] = df_all['word_in_title'] / df_all['len_of_query']
# df_all['ratio_description'] = df_all[
#     'word_in_description'] / df_all['len_of_query']
# df_all['attr'] = df_all['search_term'] + "\t" + df_all['brand']
# df_all['word_in_brand'] = df_all['attr'].map(
#     lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
# df_all['ratio_brand'] = df_all['word_in_brand'] / df_all['len_of_brand']
# df_brand = pd.unique(df_all.brand.ravel())
# d = {}
# i = 1
# for s in df_brand:
#     d[s] = i
#     i += 1
# df_all['brand_feature'] = df_all['brand'].map(lambda x: d[x])
# df_all['search_term_feature'] = df_all['search_term'].map(lambda x: len(x))

# # whether has desc or not
# df_all['isdesc'] = 1
# df_all.loc[df_all['product_description'].isnull(), 'isdesc'] = 0

# print("Calculating count feature cost--- %s seconds ---" %
#       (time.time() - start_time))


import time
start_time = time.time()

import numpy as np
import pandas as pd
#from nltk.stem.porter import *
#stemmer = PorterStemmer()
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
import re
import random
random.seed(2016)

df_train = pd.read_csv('input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('input/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('input/product_descriptions.csv')
#df_attr = codecs.open('input/product_descriptions.csv', encoding='latin1')
df_attr = pd.read_csv('input/attributes.csv', encoding="ISO-8859-1")
df_brand = df_attr[df_attr.name == "MFG Brand Name"][
    ["product_uid", "value"]].rename(columns={"value": "brand"})
num_train = df_train.shape[0]
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

stop_w = ['for', 'xbi', 'and', 'in', 'th', 'on', 'sku', 'with', 'what', 'from', 'that', 'less',
          'er', 'ing']  # 'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
          'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}


def str_stem(s):
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)  # Split words with a.A
        s = s.lower()
        s = s.replace("  ", " ")
        s = s.replace(",", "")  # could be number / segment later
        s = s.replace("$", " ")
        s = s.replace("?", " ")
        s = s.replace("-", " ")
        s = s.replace("//", "/")
        s = s.replace("..", ".")
        s = s.replace(" / ", " ")
        s = s.replace(" \\ ", " ")
        s = s.replace(".", " . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x ", " xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*", " xbi ")
        s = s.replace(" by ", " xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(
            r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(
            r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°", " degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v ", " volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  ", " ")
        s = s.replace(" . ", " ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join(
            [str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])

        s = s.lower()
        s = s.replace("toliet", "toilet")
        s = s.replace("airconditioner", "air conditioner")
        s = s.replace("vinal", "vinyl")
        s = s.replace("vynal", "vinyl")
        s = s.replace("skill", "skil")
        s = s.replace("snowbl", "snow bl")
        s = s.replace("plexigla", "plexi gla")
        s = s.replace("rustoleum", "rust-oleum")
        s = s.replace("whirpool", "whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless", "whirlpool stainless")
        return s
    else:
        return "null"


def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]", " ", str2)
    str2 = [z for z in set(str2.split()) if len(z) > 2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word) > 3:
            s1 = []
            s1 += segmentit(word, str2, True)
            if len(s) > 1:
                s += [z for z in s1 if z not in ['er',
                                                 'ing', 's', 'less'] and len(z) > 1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))


def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                # print(s[:-j],s[len(s)-j:])
                s = s[len(s) - j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i == len(st):
            r.append(st[i:])
    return r


def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word) >= 0:
            cnt += 1
    return cnt


def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

df_all['search_term'] = df_all['search_term'].map(lambda x: str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stem(x))
df_all['product_description'] = df_all[
    'product_description'].map(lambda x: str_stem(x))
df_all['brand'] = df_all['brand'].map(lambda x: str_stem(x))
print("--- Stemming: %s minutes ---" %
      round(((time.time() - start_time) / 60), 2))
df_all['product_info'] = df_all['search_term'] + "\t" + \
    df_all['product_title'] + "\t" + df_all['product_description']
print("--- Prod Info: %s minutes ---" %
      round(((time.time() - start_time) / 60), 2))
df_all['len_of_query'] = df_all['search_term'].map(
    lambda x: len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(
    lambda x: len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(
    lambda x: len(x.split())).astype(np.int64)
df_all['len_of_brand'] = df_all['brand'].map(
    lambda x: len(x.split())).astype(np.int64)
print("--- Len of: %s minutes ---" %
      round(((time.time() - start_time) / 60), 2))
df_all['search_term'] = df_all['product_info'].map(
    lambda x: seg_words(x.split('\t')[0], x.split('\t')[1]))
#print("--- Search Term Segment: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['query_in_title'] = df_all['product_info'].map(
    lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[1], 0))
df_all['query_in_description'] = df_all['product_info'].map(
    lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[2], 0))
print("--- Query In: %s minutes ---" %
      round(((time.time() - start_time) / 60), 2))
df_all['query_last_word_in_title'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[1]))
df_all['query_last_word_in_description'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[2]))
print("--- Query Last Word In: %s minutes ---" %
      round(((time.time() - start_time) / 60), 2))
df_all['word_in_title'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
df_all['ratio_title'] = df_all['word_in_title'] / df_all['len_of_query']
df_all['ratio_description'] = df_all[
    'word_in_description'] / df_all['len_of_query']
df_all['attr'] = df_all['search_term'] + "\t" + df_all['brand']
df_all['word_in_brand'] = df_all['attr'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['ratio_brand'] = df_all['word_in_brand'] / df_all['len_of_brand']
df_brand = pd.unique(df_all.brand.ravel())
d = {}
i = 1000
for s in df_brand:
    d[s] = i
    i += 3
df_all['brand_feature'] = df_all['brand'].map(lambda x: d[x])
df_all['search_term_feature'] = df_all['search_term'].map(lambda x: len(x))

# whether has desc or not
df_all['isdesc'] = 1
df_all.loc[df_all['product_description'].isnull(), 'isdesc'] = 0

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]


#df_train = pd.read_csv('input/clean_train.csv')
#df_test = pd.read_csv('input/clean_test.csv')
id_test = df_test['id']
y_train = df_train['relevance'].values


df_train = df_train.drop(['id', 'relevance', 'search_term', 'product_title',
                          'product_description', 'product_info', 'attr', 'brand'], axis=1)
df_test = df_test.drop(['id', 'relevance', 'search_term', 'product_title',
                        'product_description', 'product_info', 'attr', 'brand'], axis=1)

np.savetxt(config.path_features + 'common_word_feat_train.txt', df_train)
np.savetxt(config.path_features + 'common_word_feat_test.txt', df_test)
