import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_block import LilTransformerBlock


class LilChatBot(nn.Module):
    def __init__(
        self,
        vocab_size: int, # Number of distinct tokens in vocabulary
        seq_len: int,    # Max context length (# tokens per training example)
        d_model: int,    # "Model dimension" (i.e., width of embedding vectors)
        n_layer: int,    # Number of stacked transformer blocks
        n_head: int,     # Number of multi-attention heads
        ff_mult: int,    # How many times wider should the FF net for each
                         #   transformer block be?
        dropout: float   # Probability of zeroing out each input unit of the
                         #   submodules during training
    ):
        super().__init__()

        assert (
            d_model % n_head == 0
        ), "Embedding width must be divisible by number of heads."

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head

        # Why use nn.Embeddings instead of raw torch.Tensors? By doing it this
        # way, (1) they'll be automatically registered as "parameters" of the
        # model (and retrievable via .parameters()), (2) They'll be moved to a
        # different device if the model is moved to a different device.
        self.tok_embeds = nn.Embedding(vocab_size, d_model)
        self.pos_embeds = nn.Embedding(seq_len, d_model)

        self.trans_blocks = nn.ModuleList([
            LilTransformerBlock(d_model, n_head, ff_mult * d_model, dropout)
            for _ in range(n_layer)
        ])

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # We can do this:
        self.lm_head.weight = self.tok_embeds.weight
        # to tie the weights between the embedding and lm_head projections.
        # (This would make sense because the lm_head is essentially an
        # inverse-ish operation from embedding space back to input_id space.)


    def forward(
        self,
        input_ids: torch.Tensor, # (batch_size, seq_len)
    ):
        assert (
            input_ids.size(-1) == self.seq_len
        ), (
            f"Wrong sequence length ({input_ids.size(-1)}) passed to "
            f"LilChatBot, which was expecting {self.seq_len}."
        )

        # For now, we have no attention mask, because we're simply going to
        # divvy up our corpus into identically-sized sequences. Later, we'll
        # improve this by making each sequence correspond to a "turn" of a
        # dialogue (which will require a corpus that's labeled that way).
        tok = self.tok_embeds(input_ids)   # (batch_size, seq_len, d_model)

        # Note 1: explicitly materializing this full matrix is a little slower
        # than broadcasting it.
        # Note 2: actually, this operation isn't really necessary at all! The
        # argument we're passing is simply [0,1,2,...,seq_len-1], and passing
        # that to self.pos_embeds() will simply give us back the weights matrix
        # itself! We do it this way because of generality and clarity.
        pos_indices = torch.stack([        # (batch_size, seq_len)
            torch.arange(self.seq_len, device=input_ids.device)
        ] * input_ids.size(0))
        pos = self.pos_embeds(pos_indices) # (batch_size, seq_len, d_model)
        embeddings = tok + pos             # (batch_size, seq_len, d_model)

        transformed = embeddings
        for block in self.trans_blocks:
            transformed = block(transformed)

        return self.lm_head(transformed)


if __name__ == "__main__":
    lil = LilChatBot(1000, 50, 768, 5, 4, 4)
    input = torch.randint(0,1000,(17,50))
    print(f"input: {input.size()}")
    output = lil(input)
    print(f"output: {output.size()}")
