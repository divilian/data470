#!/usr/bin/env python
'''
Get HF models to perform text summarization on CNN/DailyMail articles, and
compute ROUGE & BLEU scores for their output.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
from pprint import pprint

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate


def generate_summary(tokenizer, model, text, min_tok=80, max_tok=300):
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



if __name__ == "__main__":
    model_name = "sshleifer/distilbart-cnn-12-6"
    model_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"
    model_name = "google/long-t5-tglobal-base"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset = load_dataset("cnn_dailymail", "3.0.0")['test']

    num = int(input(f"Choose an article number (0-{len(dataset)}) "))
    while num != -1:
        example_article = dataset[num]['article']
        example_gold_summary = dataset[num]['highlights']

        summary = generate_summary(tokenizer, model, example_article)
        print(f"\n\nArticle text:\n{example_article}")
        print(f"\n\nGold summary:\n{example_gold_summary}")
        print(f"\n\nGenerated summary:\n{summary}")

        rouge = evaluate.load("rouge")
        bleu = evaluate.load("sacrebleu")
        bertscore = evaluate.load("bertscore")
        r = rouge.compute(
            predictions=[summary],
            references=[example_gold_summary]
        )
        b = bleu.compute(
            predictions=[summary],
            references=[example_gold_summary]
        )
        bs = bertscore.compute(
            predictions=[summary],
            references=[example_gold_summary],
            model_type="bert-base-uncased"
        )
        print("ROUGE:")
        pprint({ k:r[k] for k in ['rouge1','rouge2','rougeL'] if k in r })
        print("BLEU:")
        pprint({ k:b[k] for k in ['bp','precisions','score'] if k in b })
        print("BERTScore:")
        pprint({ k:bs[k] for k in ['f1','precision','recall'] if k in bs })
        num = int(input(f"\nChoose an article number (0-{len(dataset)}) "))
