#!/usr/bin/env python
'''
Use scikit-learn to compute various metrics on a (sucky) result set from a
hypothetical multiclass classifier.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import pandas as pd
from io import StringIO
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    fbeta_score
)

labels = list("ABC")

res = pd.read_csv(StringIO(
"""
true,pred
A,B
A,B
B,B
A,C
A,C
C,B
C,C
B,B
A,C
C,A
B,B
B,B
""")
)
print(res)
pre, rec, f1, _ = precision_recall_fscore_support(
    res['true'], res['pred'], labels=labels, average=None, zero_division=0
)
acc = accuracy_score(res['true'], res['pred'])
print(f"\nAcc: {acc:.3f}")

for i,l in enumerate(labels):
    print("\n" + l + ":")
    print(f"pre: {pre[i]:.3f}")
    print(f"rec: {rec[i]:.3f}")
    print(f"f-1: {f1[i]:.3f}")

apre, arec, af1, _ = precision_recall_fscore_support(
    res['true'], res['pred'], labels=labels, average='macro', zero_division=0
)
print(f"\nAvg pre: {apre:.3f}")
print(f"Avg rec: {arec:.3f}")
print(f"Avg f-1: {af1:.3f}")

beta = 10
fscore = fbeta_score(res['true'], res['pred'], beta=beta, labels=labels,
    average="macro")
print(f"\nF_{beta}: {fscore:.3f}")
