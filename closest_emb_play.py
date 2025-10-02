#!/usr/bin/env python
'''
Interactive program to let users find the closest word embeddings to a given
word in a standard pre-trained embedding collection (like word2vec or GloVe).
Run load_embeddings.py first before running this.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import re

import gensim.downloader as api
import numpy as np

#model = api.load("glove-wiki-gigaword-300")
model = api.load("word2vec-google-news-300")

word = input("Enter a word: ")
while word != "done":
    if word not in model:
        print(f"Sorry, no embedding for {word}.")
        continue
    sims = model.most_similar(word, topn=10)
    for sim in sims:
        print(f"{sim[0]:<15}: {sim[1]:4f}")
    word = input("\nEnter a word: ")
