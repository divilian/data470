'''
Simple example to demonstrate spaCy's rule-based tokenizer.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import spacy

# Requires "python -m spacy download en_core_web_sm" the first time.
nlp = spacy.load("en_core_web_sm")

text = """
Email me at stephen@umw.edu, okay?
You can't use the gmail.com one, unfortunately.
I live in the U.S.A. I do.
Wow, spaCy is really state-of-the-art.
"""
doc = nlp(text)

# You can iterate over "doc". Each item is a spaCy Token object, with various
# properties. The most obvious one to use is ".text" which gives you the text
# of the token, but there are some other handy ones (like .is_punct in the
# example below.)

tokens = [
    token.text.lower()
    for token in doc
    if not token.is_punct and not token.is_space
]
print("The tokens are:")
print(tokens)
