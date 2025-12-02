import torch
import torch.nn as nn

from multihead_attention_block import LilMultiheadAttention
from feed_forward_block import LilFeedForwardBlock


class LilTransformerBlock(nn.Module):
    def __init__(
        self,
        # No need to initialize with seq_len, since this block will operate
        # generically on any sequence length.
        d_model: int,    # "Model dimension" (i.e., width of embedding vectors)
        n_head: int,     # Number of multi-attention heads
        ff_width: int,   # Inner width of final feed-forward transformation
        dropout: float   # Probability of zeroing out each input unit of this
                         # block's submodules during training
    ):
        super().__init__()

        self.mha = LilMultiheadAttention(d_model, n_head, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = LilFeedForwardBlock(d_model, ff_width, dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        embeddings: torch.Tensor      # (batch_size, seq_len, d_model)
    ):
        # Compute self attention for the entire token sequence. Each row of
        # "attn" now contains an embedding-like vector, for one position in the
        # sequence, that has mixed information from all the other positions in
        # the sequence.
        attn = self.mha(embeddings)

        # Include a residual connection, and normalize the result, in order to
        # stabilize inputs, fight vanishing gradients, and preserve and reuse
        # lower-level features.
        # (Note we're doing "post-norm" layer normalization, as in the original
        # Transformer paper, though "pre-norm" seems to work better for very
        # deep models like GPT-2 and beyond.)
        norm_attn = self.ln1(attn + embeddings)  # include residual connection

        # Now that self-attention has mixed information across tokens, use a FF
        # layer to process each token's vector independently. This enriches
        # token representations and boosts the model capacity.
        ff_output = self.ff(norm_attn)

        # Residual + normalization. (See comment above.)
        norm_output = self.ln2(ff_output + norm_attn)
        return norm_output
