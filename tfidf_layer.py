
import re
from collections import Counter
from typing import List, Dict, Tuple, Set

import torch
from spacy.lang.en.stop_words import STOP_WORDS

def compute_vocab(
    texts: List[str],
    k:int = 200,             # top-k words retained in vocab
    omit_stop: bool = False  # take out stopwords?
) -> Tuple[Set[str], Dict[str,int]]:
    """
    Returns two items:
        - a list of all the vocab words retained (the index number of each word
          will later be its index_id)
        - a dict with DF counts of all these words
    """

    texts = [ t.lower() for t in texts ]
    all_texts = " ".join(texts)
    all_words = re.findall(r"[A-Za-z']+", all_texts)
    counter = Counter(all_words)
    if omit_stop:
        for stop in STOP_WORDS:
            if stop in counter:
                del counter[stop]
    counter = counter.most_common(k)
    dfs = { w: sum([ w in t for t in texts ])
        for w,_ in counter }
    return [ w for w,_ in counter ], dfs

def encode_text(
    text: str,           # 1
    vocab: List[str],    # |V|
    dfs: Dict[str,int]   # |V|
) -> torch.Tensor:       # 1x|V|
    text = text.lower()
    this_docs_words = re.findall(r"[A-Za-z']+", text)
    counter = Counter(this_docs_words)
    encoded = torch.empty(len(vocab))
    for i in range(len(vocab)):
        encoded[i] = counter.get(vocab[i], 0.0)
    dfs_vec = torch.tensor([ dfs[s] for s in vocab ])
    encoded /= dfs_vec
    return encoded


def encode_all(
    texts: List[str],    # N
    vocab: List[str],    # |V|
    dfs: Dict[str,int]   # |V|
) -> torch.Tensor:       # Nx|V|
    encoded = torch.empty((len(texts),len(vocab)))
    for i,text in enumerate(texts):
        encoded[i,:] = encode_text(text, vocab, dfs)
    return encoded
    


if __name__ == "__main__":

    texts = [
        "I ate a sweet platypus",
        "go pet the sweet crocodile",
        "penny is my pet cat penny is sweet i love penny"
    ]
    vocab, dfs = compute_vocab(texts, 100, True)
    print(vocab)
    print(dfs)
    X = encode_all(texts,vocab,dfs)
    print(X)
