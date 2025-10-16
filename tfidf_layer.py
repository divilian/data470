
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
        - a set of all the vocab words retained
        - a dict with DF counts of all these words
    """

    texts = [ t.lower() for t in texts ]
    all_texts = " ".join(texts)
    x = re.findall(r"[A-Za-z']+", all_texts)
    counter = Counter(x)
    if omit_stop:
        for stop in STOP_WORDS:
            if stop in counter:
                del counter[stop]
    counter = counter.most_common(k)
    dfs = { w: sum([ w in t for t in texts ])
        for w,_ in counter }
    return { w for w,_ in counter }, dfs


if __name__ == "__main__":

    texts = [
        "I ate a sweet platypus",
        "go pet the crocodile",
        "penny is my pet cat penny is sweet i love penny"
    ]
    vocab, dfs = compute_vocab(texts, 100, True)
    print(vocab)
    print(dfs)
