#!/usr/bin/env python
'''
Interactive program that lets a user interactively type long batches of text
and inspect HF model's text summarization output.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import sys

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_summary(tokenizer, model, text, min_tok, max_tok):
    inputs = tokenizer(
        text,
#        truncation=True,
#        max_length=1024,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tok,
        min_length=min_tok
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def prompt() -> str:
    print("\nType or paste your text, then press CTRL-D on a new line:\n")
    return sys.stdin.read()


if __name__ == "__main__":

    min_tok = 120
    max_tok = 300
    model_name = "sshleifer/distilbart-cnn-12-6"
    model_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"
    model_name = "google/long-t5-tglobal-base"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    text = prompt()
    while text != 'done':
        print("Generating summary...")
        summary = generate_summary(tokenizer, model, text, min_tok, max_tok)
        print(f"\n\nGenerated summary:\n{summary}")
        text = prompt()

