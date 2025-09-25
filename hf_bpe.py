#!/usr/bin/env python
'''
Demo code that uses HuggingFace's BPE tokenizer.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
from tokenizers import ByteLevelBPETokenizer
from pprint import pprint

CORPUS = "bpe_corpus.txt"
N_MERGES = 16

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    [CORPUS],                  # list of files to train on
    vocab_size=256+3+N_MERGES, # approximate; depends on initial alphabet
)

print("The vocab is:")
vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
pprint(vocab)

tokenizer.save("tokenizer.json")

# Tokenize ("encode") and de-tokenize ("decode")
text = input("Enter text to encode (or done): ")
while text != "done":
    encoded = tokenizer.encode(text)
    print(encoded.ids)       # token ids
    print(encoded.tokens)    # token strings
    print(tokenizer.decode(encoded.ids))  # back to text
    text = input("Enter text to encode (or done): ")
