import math

import torch
import torch.nn as nn

# (Note: surprisingly, PyTorch does not have an out-of-the-box "FeedForward"
# module. For pedagogical purposes, though, we'd write our own anyway.)
class LilFeedForwardBlock(nn.Module):
    def __init__(
        self,
        # No need to initialize with seq_len, since this block will operate
        # generically on any sequence length.
        d_model: int,    # "Model dimension" (i.e., width of embedding vectors)
        ff_width: int,   # Width of the FF network to create
        dropout: float   # Probability of zeroing out each input unit during
                         #   each forward pass in training
    ):
        super().__init__()

        self.ff_width = ff_width
        self.d_model = d_model

        self.input_to_hidden = nn.Linear(d_model, ff_width)
        self.dropout_layer = nn.Dropout(dropout)
        self.hidden_to_output = nn.Linear(ff_width, d_model)
        self.activation = nn.ReLU()

    def forward(
        self,
        input_ids: torch.Tensor      # (batch_size, seq_len, d_model)
    ):
        assert input_ids.size(-1) == self.d_model, (
            f"Expected input embedding width {self.d_model}, "
            f"but got {input_ids.size(-1)} from input tensor."
        )

        hidden = self.input_to_hidden(input_ids)
        activated = self.activation(hidden)
        dropped = self.dropout_layer(activated)
        output = self.hidden_to_output(dropped)
        dropped_output = self.dropout_layer(output)
        return dropped_output
