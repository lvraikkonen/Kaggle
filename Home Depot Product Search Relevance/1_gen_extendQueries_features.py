# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import config
from utility import *
import time

#
start_time = time.time()
train = pd.read_csv(config.path_train, encoding='ISO-8859-1').fillna('')
test = pd.read_csv(config.path_test,  encoding='ISO-8859-1').fillna('')
df_pro_desc = pd.read_csv('input/product_descriptions.csv')
train = pd.merge(train, df_pro_desc, how='left', on='product_uid')
test = pd.merge(test, df_pro_desc, how='left', on='product_uid')

print("Reading data cost--- %s seconds ---" % (time.time() - start_time))

#
start_time = time.time()
print "Generating count features for training and test set..."
X1, weights, titles, queries = assemble_counts(train, m='train')
X1_test, titles_test, queries_test = assemble_counts(test, m='test')
np.savetxt(config.path_features + 'train_counts.txt', X1)
np.savetxt(config.path_features + 'test_counts.txt', X1_test)
pd.DataFrame(weights, columns=['weights']).to_csv(
    config.path_features + 'weights.csv', index=False)
pd.DataFrame(titles, columns=['titles_clean']).to_csv(
    config.path_features + 'titles_clean.csv', index=False)
pd.DataFrame(queries, columns=['queries_clean']).to_csv(
    config.path_features + 'queries_clean.csv', index=False)
pd.DataFrame(titles_test, columns=['titles_test_clean']).to_csv(
    config.path_features + 'titles_test_clean.csv', index=False)
pd.DataFrame(queries_test, columns=['queries_test_clean']).to_csv(
    config.path_features + 'queries_test_clean.csv', index=False)

print("Generating count features for training and test set cost--- %s seconds ---" %
      (time.time() - start_time))

# extended queries top 10 words
start_time = time.time()
print "Generating count features for extended query(10 words)..."
train_ext, test_ext = construct_extended_query(
    queries, queries_test, titles, titles_test, top_words=10)
X5, query_ext = assemble_counts2(train_ext.fillna(""))
X5_test, query_ext_test = assemble_counts2(test_ext.fillna(""))
np.savetxt(config.path_features + 'train_ext_counts_top10.txt', X5)
np.savetxt(config.path_features + 'test_ext_counts_top10.txt', X5_test)
tmp = pd.DataFrame(train_ext, columns=[
                   'id', 'query', 'product_title', 'product_descriptions', 'relevance'])
tmp.to_csv(config.path_features + 'train_ext_top10.csv', index=False)
tmp = pd.DataFrame(test_ext, columns=[
                   'id', 'query', 'product_title', 'product_descriptions'])
tmp.to_csv(config.path_features + 'test_ext_top10.csv', index=False)

print("Generating count features for extended training and test set cost--- %s seconds ---" %
      (time.time() - start_time))

# extended queries top 15 words
start_time = time.time()
print "Generating count features for extended query(15 words)..."
train_ext, test_ext = construct_extended_query(
    queries, queries_test, titles, titles_test, top_words=15)
X5, query_ext = assemble_counts2(train_ext.fillna(""))
X5_test, query_ext_test = assemble_counts2(test_ext.fillna(""))
np.savetxt(config.path_features + 'train_ext_counts_top15.txt', X5)
np.savetxt(config.path_features + 'test_ext_counts_top15.txt', X5_test)
tmp = pd.DataFrame(train_ext, columns=[
                   'id', 'query', 'product_title', 'product_descriptions', 'relevance'])
tmp.to_csv(config.path_features + 'train_ext_top15.csv', index=False)
tmp = pd.DataFrame(test_ext, columns=[
                   'id', 'query', 'product_title', 'product_descriptions'])
tmp.to_csv(config.path_features + 'test_ext_top15.csv', index=False)
print("Generating count features for extended training and test set cost--- %s seconds ---" %
      (time.time() - start_time))
