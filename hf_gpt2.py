#!/usr/bin/env python
'''
Demo code that uses HuggingFace's GPT2 tokenizer&model to perform text
generation.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

num_words = 80
num_choices_to_consider = 1000  # 1=greedy/Kaleb
temperature = .1

model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

txt = input('Enter text: ')
while txt != "done":
    input_ids = (tokenizer(txt, return_tensors='pt')['input_ids'].squeeze(0))

    with torch.no_grad():
        for i in range(num_words):
            output = model(input_ids=input_ids.unsqueeze(0))
            next_token_logits = output.logits[0,-1,:]

            top_vals, top_input_ids = torch.topk(
                next_token_logits / temperature, num_choices_to_consider)
            top_probs = torch.softmax(top_vals, dim=-1)

            chosen_idx = torch.multinomial(top_probs, num_samples=1)
            chosen_input_id = top_input_ids[chosen_idx]

            input_ids = torch.cat([ input_ids, chosen_input_id ])
            print("--------")
            print(tokenizer.decode(input_ids))

    print("======")
    txt = input('Enter text: ')
