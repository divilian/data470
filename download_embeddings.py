#!/usr/bin/env python
'''
Code to download standard pre-trained embedding collections using the gensim
API. This takes a while to load, but then closest_emb_play.py and
sim_emb_play.py are pretty snappy.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''

import gensim.downloader as api

# 1) Word2Vec (Google News, 300d, ~1.6GB download)
w2v = api.load("word2vec-google-news-300")

# 2) GloVe (Wikipedia+Gigaword, 300d, ~380MB download)
glove = api.load("glove-wiki-gigaword-300")

glove_100 = api.load("glove-wiki-gigaword-100")
glove_twitter_200 = api.load("glove-twitter-200")

