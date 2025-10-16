import json

import torch

from wordcount_encoder import encode_text
from train_movies import sigmoid

torch.set_printoptions(precision=2,sci_mode=False)
torch.set_printoptions(profile="default")


# Load previous weights, vocab, and doc freqs.
w = torch.load("weights.pt")
dfs_vec = torch.load("dfs_vec.pt")
with open("vocab2id.json","r",encoding="utf=8") as f:
    vocab2id = json.load(f)

review = input("Enter your review: ")
while review != "done":
    enc = encode_text(review,vocab2id,dfs_vec)
    yhat = sigmoid(enc @ w)
    print(f"yhat = {yhat:.3f}")
    review = input("Enter your review: ")
