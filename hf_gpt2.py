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
temperature = 1   # higher = more creative; lower = safer

model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

txt = input('Enter text: ')
while txt != "done":
    input_ids = (tokenizer(txt, return_tensors='pt')['input_ids'])

    with torch.no_grad():
        for i in range(num_words):
            output = model(input_ids=input_ids)
            next_token_logits = output.logits[0,-1,:]

            top_vals, top_input_ids = torch.topk(
                next_token_logits / temperature, num_choices_to_consider)
            top_probs = torch.softmax(top_vals, dim=-1)

            chosen_idx = torch.multinomial(top_probs, num_samples=1)
            new_id = top_input_ids[chosen_idx]

            input_ids = torch.cat( [ input_ids, new_id.unsqueeze(0) ], dim=1)
    print("------")
    print(tokenizer.decode(input_ids.squeeze()))

    txt = input('\nEnter text: ')
