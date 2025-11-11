#!/usr/bin/env python
'''
Manually train a movie rating classifier on the IMDB data set using a
PyTorch-designed feed-forward network.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import re
import json

from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset

torch.set_printoptions(precision=2,sci_mode=False)
torch.set_printoptions(profile="default")

def to_vec(example):
    vec = np.zeros(V, dtype=np.float32)
    for t in tok(example["text"]):
        i = stoi.get(t)
        if i is not None:
            vec[i] += 1.0
    example["vec"] = vec
    return example

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    plot_loss = False
    verbose = True
    p = 5000     # Number of most common words to retain as features.

    print("Loading IMDB data...")
    imdb = load_dataset("imdb")
    train_ds, val_ds = imdb["train"], imdb["test"]

    print("Tokenizing and computing vocabulary...")
    TOK = re.compile(r"[A-Za-z0-9_]+")

    def tok(s: str):
        return TOK.findall(s.lower())

    counter = Counter()
    for ex in imdb['train']:
        counter.update(tok(ex["text"]))

    itos = [t for t, _ in counter.most_common(p)]
    stoi = {t:i for i,t in enumerate(itos)}
    V = len(stoi)

    print("Converting strings to vectors...")
    train_vec = train_ds.map(to_vec)
    val_vec   = val_ds.map(to_vec)
    train_vec.set_format(type="torch", columns=["vec","label"])
    val_vec.set_format(type="torch", columns=["vec","label"])
    train_loader = DataLoader(train_vec, batch_size=1000, shuffle=True)
    val_loader = DataLoader(val_vec, batch_size=1000)

    print("Instantiating model...")
    model = # instantiate our model, and put it on the device.

    num_params = sum([ p.numel() for p in model.parameters() ])
    print(f"Your model has {num_params:,} parameters.")

    print("Setting up optimizer and loss function...")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    loss_func = torch.nn.CrossEntropyLoss()

    print("Beginning training loop...")
    for epoch in range(5):
        model.train()
        for batch in train_loader:
            x = batch["vec"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_func(logits, y.long())
            loss.backward()
            opt.step()
        # Compute accuracy on validation set at the end of each epoch.
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                val_x = batch["vec"].to(device)
                val_y = batch["label"].to(device)
                val_pred = model(val_x).argmax(-1)
                correct += (val_pred == val_y).sum().item()
                total += val_y.shape[0]
        print(f"After epoch {epoch+1}, validation acc: {correct/total:.3f}")

