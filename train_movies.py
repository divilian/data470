#!/usr/bin/env python
'''
Manually train a movie rating classifier on the IMDB data set using log reg.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import json
import matplotlib.pyplot as plt

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

def ce_loss(X, y, w):
    """
    Compute the mean cross-entropy loss produced by a logistic model using the
    weights w, on a data set with features X and labels y.
    """
    yhat = predict(X, w)
    return torch.sum( -(y * torch.log(yhat) + (1-y) * torch.log(1-yhat)))

def predict(X, w):
    """
    Compute the mean cross-entropy loss produced by a logistic model using the
    weights w, on a data set with features X and labels y.
    """
    yhat = sigmoid(X @ w)
    return yhat


if __name__ == "__main__":

    plot_loss = False
    verbose = True
    N = 0     # Use only this # of reviews as training data (or 0 for all).
    p = 2000     # Number of most common words to retain as TF-IDF features.

    print("Loading IMDB data...")
    imdb = load_dataset("imdb")

    print("Preparing data sets...")
    small_train = imdb['train'].shuffle(seed=123)
    if N > 0:
        small_train = small_train[:N]
    texts_train = small_train['text']
    labels_train = small_train['label']

    print("Computing vocab...")
    vocab2id, dfs_vec = compute_vocab(texts_train, p)
    label_mapper = imdb['train'].features['label'].int2str

    print("Encoding training texts...")
    X_train = encode_all(texts_train,vocab2id,dfs_vec)
    y_train = torch.tensor(labels_train)

    # Initialize weights randomly.
    w = (torch.rand(p, dtype=float) - .5).requires_grad_()

    # Set GD parameters.
    eta = .05     # Greek letter η, a.k.a. "learning rate"
    loss_delta_thresh = 0.0000001
    max_iters = 20000
    n_iter = 0

    # Prepare to plot.
    plot_vals = torch.empty((max_iters,))

    # Keep track of the mean cross-entropy loss for (1) our current weight
    # vector, and (2) last iteration's weight vector (so we can measure how
    # much better our model is getting).
    loss = ce_loss(X_train, y_train, w)
    last_loss = torch.tensor(torch.inf)

    # Loop until we hit max_iter, or until our Δ < loss_delta_thresh.
    while True:
        loss = ce_loss(X_train, y_train, w)
        plot_vals[n_iter] = loss
        loss.backward()
        with torch.no_grad():
            w -= eta * w.grad
            w.grad = None
            if verbose:
                print(f"{n_iter} iters: loss: {loss.item():4f}, "
                    f"Δ: {last_loss.item() - loss.item():.4f}")
            if (last_loss - loss).item() < loss_delta_thresh:
                break
        last_loss = loss
        n_iter += 1
        if n_iter >= max_iters:
            break

    if plot_loss:
        fig, ax = plt.subplots()
        ax.plot(range(n_iter), plot_vals[:n_iter].detach().numpy())
        ax.set_title("Cross-Entropy Loss during training")
        ax.set_xlabel("iteration")
        ax.set_ylabel("CE loss")
        plt.show()

    # Show performance on the held-out test set
    print(f"final CE loss: {ce_loss(X_train, y_train, w)} (train)")
    print(f"final CE loss: {ce_loss(X_test, y_test, w)} (test)")

    # Save our results for eval_movies.py, interact_movies.py.
    torch.save(w, "weights.pt")
    torch.save(dfs_vec, "dfs_vec.pt")
    with open("vocab2id.json","w",encoding="utf=8") as f:
        json.dump(vocab2id, f)
