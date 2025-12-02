import argparse
import os
from typing import List
import json

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tokenizers import Tokenizer, ByteLevelBPETokenizer

from lil_chatbot import LilChatBot


# Some training tips:
#
# - $ nvidia-smi will tell you what % of the GPU memory you're using. It kinda
# makes sense to increase the batch size as much as possible until you hit this
# limit (although it does affect learning somewhat, stabilizing gradients and
# what-not).
#
# - the higher you make the batch size, the higher the training rate you want
# (larger batch sizes provide more stable gradient estimates, reducing noise,
# which allow for larger updates without diverging). This is called the linear
# scaling rule: "If you multiply your batch size by k, you can often multiply
# your learning rate by k as well."


_tokenizer = None
def get_tokenizer(
    corpus_name:str,   # Name of file in ~/corpora (w/o ".txt" extension).
    model_name:str,    # Name of file in local dir to store tokenizer.
    vocab_size:int = 2000,
    # Strings that should be specially treated as individual tokens.
    special_tokens:list = []   
):
    """
    Reload, or train from scratch on the corpus passed, a BPE tokenizer.
    """
    global _tokenizer
    if _tokenizer is None:
        tok_file = model_name + "_tok.json"
        if os.path.isfile(tok_file):
            _tokenizer = Tokenizer.from_file(tok_file)
        else:
            # We're not using a pretrained tokenizer because its vocab will be
            # too large and not aligned with our corpus.
            _tokenizer = ByteLevelBPETokenizer()
            full_corpus_name = "/home/stephen/corpora/" + corpus_name + ".txt"
            assert (
                os.path.isfile(full_corpus_name)
            ) , f"Non-existent corpus file {full_corpus_name}!"
            _tokenizer.train(
                files=full_corpus_name,
                vocab_size=vocab_size,
                min_frequency=2,
                special_tokens=special_tokens
            )
            _tokenizer.save(tok_file)
    return _tokenizer


def to_seqs(tokens:List[int], seq_size:int):
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


def to_batches(seqs:torch.Tensor, batch_size:int, shuffle:bool = False):
    """
    Given a 2-d Tensor of sequences, assemble and return a 3-d Tensor of size
    ((n - seq_size + 1) // batch_size, batch_size, seq_size). The first
    dimension of this Tensor is the batch number, the second is the sequence
    number, and the second is the token within the sequence.

    Example: if seqs is
        [[1,2,3,4],
         [2,3,4,5],
         [3,4,5,6],
         [4,5,6,7],
         [5,6,7,8]]
    and batch_size is 2, this returns:
        [[[1,2,3,4],
          [2,3,4,5]],
         [[3,4,5,6],
          [4,5,6,7]]]

    If shuffle=True, shuffle the list of sequences (not the sequences
    themselves!) before batching.
    """
    if shuffle:
        seqs = seqs[torch.randperm(seqs.size(0))]
    num_full_batches = seqs.size(0) // batch_size
    trimmed = seqs[:num_full_batches * batch_size]  # Trim extra rows
    return trimmed.reshape(num_full_batches, batch_size, seqs.size(1))


def compute_loss(model, seqs, batch_size, loss_fn, device):
    """
    Compute average loss for the sequences passed. Note this happens in eval
    mode, and all at once, for (say) a whole epoch.
    """
    model.eval()
    batches = to_batches(seqs, batch_size)
    total_loss = 0.0
    with torch.no_grad():
        for b in range(batches.size(0)):
            input_ids = batches[b][:, :-1].to(device)
            gold_ids = batches[b][:, 1:].to(device)
            logits = model(input_ids)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                gold_ids.reshape(-1)
            )
            total_loss += loss.item()
    avg_loss = total_loss / batches.size(0)
    return avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a LilChatBot.")
    parser.add_argument(
        "corpus_name",
        type=str,
        help="Name of file in /home/stephen/corpora (without .txt extension)"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Filename under which to store trained model"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=2000,
        help="Number of vocab words tokenizer will track"
    )
    parser.add_argument(
        "-d","--dialog",
        action="store_true",
        help="Is source text annotated with dialog markers <A>, <EOT>, etc?"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help='"Model dimension" (width of embeddings)'
    )
    parser.add_argument(
        "--ff-mult",
        type=int,
        default=4,
        help="Width (in d_model's) of final FF net in each transformer block"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=.1,
        help="Dropout rate for all layers"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of multi-attention heads"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=896,
        help="Batch size for training"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=50,
        help="Length of each sequence in a training batch"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previous model if exists?"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot learning curve?"
    )

    args = parser.parse_args()

    source_text = "/home/stephen/corpora/" + args.corpus_name + ".txt"
    assert (
        os.path.isfile(source_text)
    ), f"No such file {source_text}!"

    assert (
        args.overwrite or not os.path.exists(args.model_name)
    ), f"Model {args.model_name} already exists! (Consider --overwrite flag.)"

    device = torch.device('cuda')
    lil = LilChatBot(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout
    ).to(device)

    tok = get_tokenizer(
        args.corpus_name,
        args.model_name,
        args.vocab_size,
        ["<A>","<B>","<EOT>","<EOD>"] if args.dialog else []
    )
    with open(source_text, "r", encoding="utf-8") as f:
        tokens = tok.encode(f.read()).ids

    # Create sequences of length seq_len+1 since we're going to be removing the
    # last token in each sequence before passing them to the model (and
    # comparing the predicted outputs to those sequences with the first token
    # in each sequence removed).
    seqs = to_seqs(tokens, args.seq_len+1)       # approx (n, seq_len+1)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(lil.parameters(), lr=args.eta)

    # Train/validation split
    val_ratio = 0.2
    num_val = int(len(seqs) * val_ratio)
    num_train = len(seqs) - num_val
    train_seqs, val_seqs = torch.utils.data.random_split(
        seqs,
        [num_train, num_val]
    )
    train_seqs = torch.stack([seqs[i] for i in train_seqs.indices])
    val_seqs = torch.stack([seqs[i] for i in val_seqs.indices])

    # (For plotting.)
    train_losses = []
    val_losses = []

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}...")
        lil.train()  # Put model in "training mode" so dropout is turned on
        epoch_loss = 0.0
        batches = to_batches(
            seqs,
            args.batch_size,
            shuffle=True
        )   # ((n-seq_len+1)//batch_size, batch_size, seq_len+1)
        for b in tqdm(range(batches.size(0))):
            # Only move one batch at a time to the GPU!
            input_ids = batches[b][:,:-1].to(device)
            gold_ids = batches[b][:,1:].to(device)

            model_logits = lil(input_ids)
            batch_loss = loss_fn(
                model_logits.reshape(-1, args.vocab_size),
                gold_ids.reshape(-1)
            )
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            epoch_loss += batch_loss.item()

        print("Computing epoch losses...")
        train_losses.append(
            compute_loss(
                lil,
                train_seqs,
                args.batch_size,
                loss_fn,
                device
            )
        )
        val_losses.append(
            compute_loss(
                lil,
                val_seqs,
                args.batch_size,
                loss_fn,
                device
            )
        )


    # You *can* save the full model like this:
    #   torch.save(model, "lilchatbot_full.pt")
    # But it's less portable and more brittle - it depends on Python and
    # PyTorch versions, and requires the exact same class definition when
    # loading. For most use cases, saving the state_dict() is safer and
    # cleaner.
    print("Saving model...")
    torch.save(lil.state_dict(), args.model_name + ".pt")

    # And we'll save all our hyperparameters so we can recreate at load time.
    with open(args.model_name + "_config.json", "w") as f:
        json.dump({
            "vocab_size": args.vocab_size,
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "n_layer": args.num_layers,
            "n_head": args.num_heads,
            "ff_mult": args.ff_mult,
            "dropout": args.dropout
        }, f)

    if args.plot:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
        ax.plot(train_losses, label="train", color="blue")
        ax.plot(val_losses, label="val", color="orange")
        ax.set_xlabel("epoch")
        ax.set_ylabel("Average CE loss")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        fig.suptitle(f"Learning curve for {args.model_name}")
        fig.tight_layout()
        plt.savefig(args.model_name + ".png")
        print(f"(Learning curve saved to {args.model_name}.png)")
