# -*- coding: utf-8 -*-
# utility methods
from __future__ import division
import numpy as np
import pandas as pd
import backports.lzma as lzma

from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import text
from difflib import SequenceMatcher as seq_matcher
from itertools import combinations_with_replacement
from sklearn.preprocessing import MinMaxScaler
import re
from collections import Counter
from nltk.stem.snowball import SnowballStemmer


def construct_extended_query(queries, queries_test, titles, titles_test, top_words=10):
    y = pd.read_csv('input/train.csv').relevance.values

    stop_words = text.ENGLISH_STOP_WORDS
    pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')

    train = pd.read_csv('input/train.csv', encoding='ISO-8859-1')
    test = pd.read_csv('input/test.csv', encoding='ISO-8859-1')
    df_pro_desc = pd.read_csv('input/product_descriptions.csv')

    train = pd.merge(train, df_pro_desc, how='left', on='product_uid')
    test = pd.merge(test, df_pro_desc, how='left', on='product_uid')

    data = []
    query_ext_train = np.zeros(len(train)).astype(np.object)
    query_ext_test = np.zeros(len(test)).astype(np.object)
    for q in np.unique(queries):
        q_mask = queries == q
        q_test = queries_test == q

        titles_q = titles[q_mask]
        y_q = y[q_mask]

        good_mask = y_q == 3
        titles_good = titles_q[good_mask]
        ext_q = str(q)
        for item in titles_good:
            ext_q += ' ' + str(item)
        ext_q = pattern.sub('', ext_q)
        c = [word for word, it in Counter(
            ext_q.split()).most_common(top_words)]
        c = ' '.join(c)
        data.append([q, ext_q, c])
        query_ext_train[q_mask] = c
        query_ext_test[q_test] = c

    train['query'] = query_ext_train
    test['query'] = query_ext_test
    train['product_title'] = titles
    test['product_title'] = titles_test
    return train, test


def compression_distance(x, y, l_x=None, l_y=None):
    if x == y:
        return 0.0
    x_b = x.encode('utf-8')
    y_b = y.encode('utf-8')
    if l_x is None:
        l_x = len(lzma.compress(x_b))
        l_y = len(lzma.compress(y_b))
    l_xy = len(lzma.compress(x_b + y_b))
    l_yx = len(lzma.compress(y_b + x_b))
    dist = (min(l_xy, l_yx) - min(l_x, l_y)) / max(l_x, l_y)
    return dist


# def correct_string(s):
#     s = s.lower()
#     s = s.replace("hardisk", "hard drive")
#     s = s.replace("extenal", "external")
#     s = s.replace("soda stream", "sodastream")
#     s = s.replace("fragance", "fragrance")
#     s = s.replace("16 gb", "16gb")
#     s = s.replace("32 gb", "32gb")
#     s = s.replace("500 gb", "500gb")
#     s = s.replace("2 tb", "2tb")
#     s = s.replace("shoppe", "shop")
#     s = s.replace("refrigirator", "refrigerator")
#     s = s.replace("assassinss", "assassins")
#     s = s.replace("harleydavidson", "harley davidson")
#     s = s.replace("harley-davidson", "harley davidson")
#     s = s.replace("rigid", "RIDGID")
#     s = s.replace("grills-gas", "grills gas")
#     s = s.replace("&amp;", "&")
#     s = s.replace("&#39;", "'")
#     s = s.replace("A/C", "Air Conditioner")
#     s = s.replace("condit", "Air Conditioner")
#     s = s.replace("ac", "Air Conditioner")

#     s = s.replace("'", "in.")
#     s = s.replace("inches", "in.")
#     s = s.replace("inch", "in.")
#     s = s.replace(" in ", "in. ")
#     s = s.replace(" in.", "in.")
#     s = s.replace("''", "ft.")
#     s = s.replace(" feet ", "ft. ")
#     s = s.replace("feet", "ft.")
#     s = s.replace("foot", "ft.")
#     s = s.replace(" ft ", "ft. ")
#     s = s.replace(" ft.", "ft.")

#     s = s.replace(" pounds ", "lb. ")
#     s = s.replace(" pound ", "lb. ")
#     s = s.replace("pound", "lb.")
#     s = s.replace(" lb ", "lb. ")
#     s = s.replace(" lb.", "lb.")
#     s = s.replace(" lbs ", "lb. ")
#     s = s.replace("lbs.", "lb.")
#     s = s.replace(" x ", " xby ")
#     s = s.replace("*", " xby ")
#     s = s.replace(" by ", " xby")
#     s = s.replace("x0", " xby 0")
#     s = s.replace("x1", " xby 1")
#     s = s.replace("x2", " xby 2")
#     s = s.replace("x3", " xby 3")
#     s = s.replace("x4", " xby 4")
#     s = s.replace("x5", " xby 5")
#     s = s.replace("x6", " xby 6")
#     s = s.replace("x7", " xby 7")
#     s = s.replace("x8", " xby 8")
#     s = s.replace("x9", " xby 9")
#     s = s.replace("0x", "0 xby ")
#     s = s.replace("1x", "1 xby ")
#     s = s.replace("2x", "2 xby ")
#     s = s.replace("3x", "3 xby ")
#     s = s.replace("4x", "4 xby ")
#     s = s.replace("5x", "5 xby ")
#     s = s.replace("6x", "6 xby ")
#     s = s.replace("7x", "7 xby ")
#     s = s.replace("8x", "8 xby ")
#     s = s.replace("9x", "9 xby ")

#     s = s.replace(" sq ft", "sq.ft. ")
#     s = s.replace("sq ft", "sq.ft. ")
#     s = s.replace("sqft", "sq.ft. ")
#     s = s.replace(" sqft ", "sq.ft. ")
#     s = s.replace("sq. ft", "sq.ft. ")
#     s = s.replace("sq ft.", "sq.ft. ")
#     s = s.replace("sq feet", "sq.ft. ")
#     s = s.replace("square feet", "sq.ft. ")

#     s = s.replace(" gallons ", "gal. ")
#     s = s.replace(" gallon ", "gal. ")
#     s = s.replace("gallons", "gal.")
#     s = s.replace("gallon", "gal.")
#     s = s.replace(" gal ", "gal. ")
#     s = s.replace(" gal", "gal.")
#     s = s.replace("ounces", "oz.")
#     s = s.replace("ounce", "oz.")
#     s = s.replace(" oz.", "oz. ")
#     s = s.replace(" oz ", "oz. ")
#     s = s.replace("centimeters", "cm.")
#     s = s.replace(" cm.", "cm.")
#     s = s.replace(" cm ", "cm. ")

#     s = s.replace("milimeters", "mm.")
#     s = s.replace(" mm.", "mm.")
#     s = s.replace(" mm ", "mm. ")

#     s = s.replace("°", "deg. ")
#     s = s.replace("degrees", "deg. ")
#     s = s.replace("degree", "deg. ")

#     s = s.replace("volts", "volt. ")
#     s = s.replace("volt", "volt. ")
#     s = s.replace("watts", "watt. ")
#     s = s.replace("watt", "watt. ")
#     s = s.replace("ampere", "amp. ")
#     s = s.replace("amps", "amp. ")
#     s = s.replace(" amp ", "amp. ")

#     s = s.replace("whirpool", "whirlpool")
#     s = s.replace("whirlpoolga", "whirlpool")
#     s = s.replace("whirlpoolstainless", "whirlpool stainless")

#     s = s.replace(" amps ", " Amp ")  # character
#     s = s.replace(" amps", "Amp ")  # whole word
#     s = s.replace("&#39;", "'")
#     return s

def correct_string(s):
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
    s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?",
               r"\1sq.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?",
               r"\1cu.ft. ", s)
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

#stemmer = PorterStemmer()
stemmer = SnowballStemmer('english')

# Stemming functionality


class stemmerUtility(object):
    # Stemming functionality

    @staticmethod
    def stemPorter(review_text):
        porter = PorterStemmer()
        preprocessed_docs = []
        for doc in review_text:
            final_doc = []
            for word in doc:
                final_doc.append(porter.stem(word))
                # final_doc.append(wordnet.lemmatize(word)) #note that
                # lemmatize() can also takes part of speech as an argument!
            preprocessed_docs.append(final_doc)
        return preprocessed_docs


def assemble_counts(train, m='train'):
    X = []
    titles = []
    queries = []
    weights = []
    train['isdesc'] = 1  # Description present flag
    train.loc[train['product_description'].isnull(), 'isdesc'] = 0

    for i in range(len(train.id)):
        query = correct_string(train['search_term'][i].lower())
        title = correct_string(train.product_title[i].lower())

        query = (" ").join(
            [z for z in BeautifulSoup(query, "lxml").get_text(" ").split(" ")])
        title = (" ").join(
            [z for z in BeautifulSoup(title, "lxml").get_text(" ").split(" ")])

        query = text.re.sub("[^a-zA-Z0-9]", " ", query)
        title = text.re.sub("[^a-zA-Z0-9]", " ", title)

        query = (" ").join([stemmer.stem(z) for z in query.split(" ")])
        title = (" ").join([stemmer.stem(z) for z in title.split(" ")])

        query = " ".join(query.split())
        title = " ".join(title.split())

        dist_qt = compression_distance(query, title)
        dist_qt2 = 1 - seq_matcher(None, query, title).ratio()

        query_len = len(query.split())
        title_len = len(title.split())
        isdesc = train.isdesc[i]

        tmp_title = title
        word_counter_qt = 0
        lev_dist_arr = []
        for q in query.split():
            lev_dist_q = []
            for t in title.split():
                lev_dist = seq_matcher(None, q, t).ratio()
                if lev_dist > 0.9:
                    word_counter_qt += 1
                    # tmp_title += ' '+q # add such words to title to increase
                    # their weights in tfidf
                lev_dist_q.append(lev_dist)
            lev_dist_arr.append(lev_dist_q)
        last_word_in = 0
        for t in title.split():
            lev_dist = seq_matcher(None, query.split()[-1], t).ratio()
            if lev_dist > 0.9:
                last_word_in = 1
        lev_max = 0
        for item in lev_dist_arr:
            lev_max_q = max(item)
            lev_max += lev_max_q
        lev_max = 1 - lev_max / len(lev_dist_arr)
        word_counter_qt_norm = word_counter_qt / query_len

        X.append([query_len, title_len, isdesc, word_counter_qt, dist_qt,
                  dist_qt2, lev_max, last_word_in, word_counter_qt_norm])
        titles.append(tmp_title)
        queries.append(query)
        if m == 'train':
            weights.append(1 / (float(train["relevance"][i]) + 1.0))
    X = np.array(X).astype(np.float)
    if m == 'train':
        return X, np.array(weights).astype(np.float), np.array(titles), np.array(queries)
    else:
        return X, np.array(titles), np.array(queries)


def assemble_counts2(train):
    X = []
    queries = []

    for i in range(len(train.id)):
        query = train['search_term'][i]
        title = train.product_title[i]

        dist_qt = compression_distance(query, title)
        dist_qt2 = 1 - seq_matcher(None, query, title).ratio()

        query_len = len(query.split())

        lev_dist_arr = []
        word_rank_list = []
        word_q_ind = 0
        word_counter_qt = 0
        for q in query.split():
            word_q_ind += 1
            lev_dist_q = []
            for t in title.split():
                lev_dist = seq_matcher(None, q, t).ratio()
                if lev_dist > 0.9:
                    word_counter_qt += 1
                    word_rank_list.append(word_q_ind)
                    # tmp_title += ' '+q # add such words to title to increase
                    # their weights in tfidf
                lev_dist_q.append(lev_dist)
            lev_dist_arr.append(lev_dist_q)
        if word_counter_qt == 0:
            maxrank = 0
        else:
            maxrank = 26 - min(word_rank_list)

        lev_max = 0
        for item in lev_dist_arr:
            lev_max_q = max(item)
            lev_max += lev_max_q
        lev_max = 1 - lev_max / len(lev_dist_arr)
        word_counter_qt_norm = word_counter_qt / query_len

        X.append([word_counter_qt, dist_qt, dist_qt2,
                  lev_max, word_counter_qt_norm, maxrank])
        queries.append(query)

    X = np.array(X).astype(np.float)

    return X, np.array(queries)
