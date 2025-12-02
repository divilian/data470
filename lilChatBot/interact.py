import json
import argparse
import os
from typing import List

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.optim as optim
from tokenizers import Tokenizer

from lil_chatbot import LilChatBot


def to_seq(tokens:List[int], seq_size:int):
    """
    Given a one-dimensional list of n tokens, split and return a Tensor of size
    (n - seq_size + 1, seq_size). The first dimension of this Tensor is the
    sequence number, and the second is the token within the sequence.

    Example: if tokens is [1,2,3,4,5,6,7,8] and seq_size is 4, this returns:
    [[1,2,3,4],
     [2,3,4,5],
     [3,4,5,6],
     [4,5,6,7],
     [5,6,7,8]]
    """
    tokens = torch.tensor(tokens, dtype=torch.long)
    return tokens.unfold(0, seq_size, 1)


def parse_args(description):
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "model_name",
        type=str,
        help="Filename from which to load trained model (and tokenizer)"
    )
    parser.add_argument(
        "--temp",
        type=float,
        help="Decoding temperature (high = more random, low = more confident)",
        default=1.0
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Sample from only this many (most probable) tokens"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Generate no more than this number of tokens per prompt"
    )

    return parser.parse_args()


def load_model(model_name:str):
    """
    Instantiate a previously-trained-and-saved LilChatBot whose base filename 
    is passed. Returns a tuple of:

    0. The LilChatBot object
    1. A dict containing the stored model configuration
    2. The tokenizer that was saved along with this model
    """
    state_dict = model_name + ".pt"
    config_file = model_name + "_config.json"
    tok_config = model_name + "_tok.json"
    assert (
        os.path.isfile(state_dict)
    ), f"No such model file {state_dict}!"
    assert (
        os.path.isfile(config_file)
    ), f"No such config file {config_file}!"
    assert (
        os.path.isfile(tok_config)
    ), f"No such tokenizer file {tok_config}!"

    # Load the tokenizer that was used when training the model.
    tok = Tokenizer.from_file(tok_config)

    # Load the model configuration, and instantiate an (uninitialized) model.
    with open(config_file, "r") as f:
        model_config = json.load(f)
    lil = LilChatBot(
        vocab_size=model_config['vocab_size'],
        seq_len=model_config['seq_len'],
        d_model=model_config['d_model'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        ff_mult=model_config['ff_mult'],
        dropout=model_config['dropout']
    )

    # Load the model state ("state_dict") from disk.
    lil.load_state_dict(torch.load(state_dict, weights_only=True))

    # Put in evaluation mode, to disable dropout.
    lil.eval()

    return lil, model_config, tok


def interact(
    model:LilChatBot,     # Model used to generate responses
    model_config:dict,    # Model parameters (stored after training)
    tok:Tokenizer,        # Tokenizer (stored after training)
    k:int,                # Consider only this many tokens when decoding
    temperature:float,    # Decoding temp; low=stable, high=risky         
    max_tokens:int,       # Do not generate more than this # of tokens
    pre_prompt:str = "",  # Model-specific text to insert before prompt
    post_prompt:str = "", # Model-specific text to insert after prompt
    end_trigger:str = "", # Text which, if generated, will end turn
    separator:str = ""    # Text to insert between prompt and response
):
    """
    Begin an (endless) interactive loop, in which the user types a prompt and
    the model gives output.
    """
    end_ids = tok.encode(end_trigger).ids
    with torch.no_grad():
        prompt = input("Enter prompt: ")
        while prompt != "done":
            prompt_enh = pre_prompt + prompt + post_prompt
            ids = tok.encode(prompt_enh).ids
            prompt_len = len(ids)
            for _ in range(max_tokens):
                if len(ids) < model_config['seq_len']:
                    padding = [0] * (model_config['seq_len'] - len(ids))
                    # Pad to the left.
                    input_ids = padding + ids
                else:
                    # Truncate the beginning.
                    input_ids = ids[-model_config['seq_len']:]
                ids_t = torch.tensor([input_ids], dtype=torch.long)
                logits = model(ids_t)           # (1, seq_len, vocab_size)
                next_tk_logits = logits[0, -1]  # (vocab_size,)
                next_tk_logits /= temperature
        
                values, indices = torch.topk(next_tk_logits, k)
                next_tk_logits[next_tk_logits < values[-1]] = float('-inf')
                probs = torch.softmax(next_tk_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                ids.append(next_token_id)
                if (
                    _ > 1 and    # (Overlook premature ending.)
                    end_trigger and
                    ids[-len(end_ids):] == end_ids
                ):
                    ids = ids[:-len(end_ids)]
                    break

            print(prompt + separator + tok.decode(ids[prompt_len:]))
            prompt = input("Enter prompt: ")


if __name__ == "__main__":

    args = parse_args("Interact with a LilChatBot.")

    lil, model_config, tok = load_model(args.model_name)

    interact(
        lil,
        model_config,
        tok,
        args.k,
        args.temp,
        args.max_tokens
    )
