
import re
from collections import Counter
from typing import List, Dict, Tuple, Set

import torch
from spacy.lang.en.stop_words import STOP_WORDS

def compute_vocab(
    texts: List[str],
    k:int = 200,             # top-k words retained in vocab
) -> Tuple[Dict[str,int], torch.Tensor]:
    """
    Given a list of texts, figure out all the unique vocab words to use, assign
    each one an "input_id", and compute the document frequency of each one.

    Returns two items:
        - a dict of all the vocab words retained, and their assigned input_id
        - a tensor with the DF counts by input id
    """
    docs_tokens = [
        re.findall(r"[A-Za-z']+", t.lower())
        for t in texts
    ]

    counter = Counter()
    for toks in docs_tokens:
        counter.update(tok for tok in toks if tok not in STOP_WORDS)

    most_common = counter.most_common(k)
    vocab = [ w for w,f in most_common ]
    vocab2id = { w: i for i, w in enumerate(vocab) }

    dfs = Counter()
    for toks in docs_tokens:
        unique = set(tok for tok in toks if tok not in STOP_WORDS)
        dfs.update(w for w in unique if w in vocab2id)

    dfs_vec = torch.tensor([dfs[w] for w in vocab2id.keys()], dtype=float)

    return vocab2id, dfs_vec


def encode_text(
    text: str,
    vocab2id: Dict[str,int], # |V|
    dfs_vec: torch.Tensor    # |V|
) -> torch.Tensor:           # |V|
    """
    Given a text, along with a mapping from vocab word to input_id
    and a vector of doc freqs, return a |V|-dimensional vector of its
    TF-IDF counts, where |V| is the number of vocabulary words used.
    """
    this_docs_words = re.findall(r"[A-Za-z']+", text.lower())
    ids = [vocab2id[w] for w in this_docs_words if w in vocab2id]
    if not ids:
        return torch.zeros(len(vocab2id), dtype=float)
    counts = torch.bincount(torch.tensor(ids),
        minlength=len(vocab2id)).to(float)
    return counts / dfs_vec    # Compute TF-IDF


def encode_all(
    texts: List[str],        # N     the whole enchilada (all texts)
    vocab2id: Dict[str,int], # |V|   maps vocab word to input_id
    dfs_vec: torch.Tensor    # |V|   indexed by input_id
) -> torch.Tensor:           # Nx|V|
    """
    Given a list of N texts, along with a mapping from vocab word to input_id
    and a vector of doc freqs, return an Nx|V| tensor of TF-IDF counts, where
    |V| is the number of vocabulary words used, and each row is one encoded
    text.
    """
    encoded = torch.empty((len(texts),len(vocab2id)), dtype=float)
    for i,text in enumerate(texts):
        if i % 100 == 0: print(f"{i}/{len(texts)}...")
        encoded[i,:] = encode_text(text, vocab2id, dfs_vec)
    return encoded
    


if __name__ == "__main__":

    texts = [
        "I ate a sweet platypus",
        "go pet the sweet crocodile",
        "penny is my pet cat penny is sweet i love penny"
    ]
    vocab2id, dfs_vec = compute_vocab(texts, 100)
    print(vocab2id)
    print(dfs_vec)
    X = encode_all(texts,vocab2id,dfs_vec)
    print(X)
