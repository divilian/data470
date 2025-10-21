#!/usr/bin/env python
'''
Evaluate a movie rating classifier trained using train_movies.py.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import json
import matplotlib.pyplot as plt

import pandas as pd
import torch
from datasets import load_dataset, Dataset

from wordcount_encoder import compute_vocab, encode_all

torch.set_printoptions(precision=2,sci_mode=False)
torch.set_printoptions(profile="default")


def sigmoid(z):
    """
    Convert a logit (from -∞ to ∞) to a probability (from 0 to 1).
    """
    return 1 / (1 + torch.exp(-z))

def predict(X, w):
    """
    Compute the mean cross-entropy loss produced by a logistic model using the
    weights w, on a data set with features X and labels y.
    """
    yhat = sigmoid(X @ w)
    return yhat


if __name__ == "__main__":

    # Load previous weights, vocab, and doc freqs.
    w = torch.load("weights.pt")
    dfs_vec = torch.load("dfs_vec.pt")
    with open("vocab2id.json","r",encoding="utf=8") as f:
        vocab2id = json.load(f)

    print("Loading IMDB data...")
    imdb = load_dataset("imdb")

    print("Preparing data sets...")
    test_set = imdb['test'].shuffle(seed=123)
    texts_test = test_set['text']
    labels_test = test_set['label']

    label_mapper = imdb['test'].features['label'].int2str

    print("Encoding test texts...")
    X_test = encode_all(texts_test,vocab2id,dfs_vec)
    y_test = torch.tensor(labels_test)

    yhat = predict(X_test, w)
    accuracy = (
        ((y_test == 1) & (yhat > .5)) |
        ((y_test == 0) & (yhat <= .5))
    ).sum() / len(y_test)

    cm = pd.crosstab(yhat > .5, y_test == 1,
        rownames=['pred'],
        colnames=['gold'])
    print(cm)
    pos_precision = ((yhat > .5) & (y_test == 1)).sum() / ((yhat > .5).sum())
    neg_precision = ((yhat <= .5) & (y_test == 0)).sum() / ((yhat <= .5).sum())
    pos_recall = ((yhat > .5) & (y_test == 1)).sum() / ((y_test == 1).sum())
    neg_recall = ((yhat <= .5) & (y_test == 0)).sum() / ((y_test == 0).sum())
    pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall)
    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Pos-precision: {pos_precision:.3f}")
    print(f"Neg-precision: {neg_precision:.3f}")
    print(f"Pos-recall: {pos_recall:.3f}")
    print(f"Neg-recall: {neg_recall:.3f}")
    print(f"Pos-f1: {pos_f1:.3f}")
    print(f"Neg-f1: {neg_f1:.3f}")

