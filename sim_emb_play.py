#!/usr/bin/env python
'''
Interactive program to let users compute the similarity between pairs of word
embeddings in a standard pre-trained embedding collection (like word2vec or
GloVe). Run load_embeddings.py first before running this.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import re

import gensim.downloader as api
import numpy as np

#model = api.load("glove-wiki-gigaword-300")
model = api.load("word2vec-google-news-300")

words = input("Enter two words, comma-separated: ")
while words != "done":
    words = re.split(r",\s*",words)
    if words[0] not in model:
        print(f"Sorry, no embedding for {words[0]}.")
        words = input("\nEnter two words, comma-separated: ")
        continue
    if words[1] not in model:
        print(f"Sorry, no embedding for {words[1]}.")
        words = input("\nEnter two words, comma-separated: ")
        continue
    embed1 = model.get_vector(words[0])
    embed2 = model.get_vector(words[1])
    sim = embed1 @ embed2 / np.linalg.norm(embed1) / np.linalg.norm(embed2)
    print(f"Similarity: {sim:.4f}")
    words = input("\nEnter two words, comma-separated: ")
