import math

import torch
import torch.nn as nn

# (Note: PyTorch does provide a nn.MultiheadAttention class, but for
# pedagogical purposes we're writing our own.)
class LilMultiheadAttention(nn.Module):
    def __init__(
        self,
        # No need to initialize with seq_len, since this block will operate
        # generically on any sequence length.
        d_model: int,    # "Model dimension" (i.e., width of embedding vectors)
        n_head: int,     # Number of multi-attention heads
        dropout: float   # Probability of zeroing out each input unit during
                         #   each forward pass in training
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model

        # Note 1: Why use nn.Linears instead of raw torch.Tensors? By doing it
        # this way, (1) they'll be automatically registered as trainable
        # parameters of the model (and retrievable via .parameters()), (2)
        # They'll be moved to a different device if the model is moved to a
        # different device.
        #
        # Note 2: Why bias=False? GPT-style models often omit bias in Q/K/V
        # projections (though this isn't mandatory).
        #
        # Note 3: Why projections of shape (d_model, d_model) instead of
        # (n_head, n_head)? Because the Q/K/V projections don't represent an
        # *individual* head's attention information, but rather "all the heads
        # at once." We'll divvy up the results after the projection, for
        # efficiency reasons.
        # Each head ("head #0", "head #1", etc.) operates on a specific *slice*
        # of the Q/K/V outputs - these slices correspond to *segments of each
        # one's output dimension*, not the weight matrix itself.
        #
        # Note 4: The Q and K projections must end up being the same size,
        # because every token's Q projection will be dot-product-ed with every
        # token's K projection, and dot-product is only defined for same-length
        # vectors. Why make that size be head_dim (which is d_model/n_head) in
        # particular, though? Answer: it doesn't have to be, and some other
        # models play with this, but GPT-style models have traditionally
        # standardized Q and K to have that shape. (This is mainly for
        # parallelism and code simplicity, not necessity.)
        #
        # Note 5: The V projection doesn't need to be the same size as either Q
        # or K, since it won't be dot-producted with them (but simply selected
        # and linearly combined, based on the Q·K products). So why make its
        # per-head output also be head_dim? Answer: same. Tradition. And some
        # models do play with different sizes here.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Add a final linear projection for the output, so that this module can
        # recombine the outputs of its attention heads in a learnable way.
        self.output_proj = nn.Linear(d_model, d_model, bias=True)

        # Perform random dropout to discourage overfitting.
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        embeddings: torch.Tensor      # (batch_size, seq_len, d_model)
    ):
        # Make a note of all the sizes for this forward pass.
        batch_size = embeddings.size(0)
        seq_len = embeddings.size(1)
        d_model = embeddings.size(-1)
        head_width = d_model // self.n_head

        assert d_model == self.d_model, (
            f"Expected input embedding width {self.d_model}, "
            f"but got {d_model} from input tensor."
        )
        assert d_model % self.n_head == 0, (
            f"Expected input embedding width (got {d_model}) "
            f"to be divisible by number of heads ({self.n_head})."
        )

        # Project the inputs into the three canonical MHA projections. (Recall
        # that if A is a nn.Linear module with n input features and p output
        # features, then A.weight has shape (p, n). Calling A(B) for a tensor B
        # of shape (..., m, n) computes B @ A.weight.T + A.bias, yielding a
        # result that is (..., m, p).)
        Q = self.q_proj(embeddings)
        K = self.k_proj(embeddings)
        V = self.v_proj(embeddings)

        # In order to perform efficient multi-head attention, we must split our
        # projections into heads, which means reshaping tensors of this shape:
        #   (batch_size, seq_len, d_model)
        # into this shape:
        #   (batch_size, n_head, seq_len, head_width)
        q_heads = self._split_into_heads(Q)
        k_heads = self._split_into_heads(K)
        v_heads = self._split_into_heads(V)

        # attn_scores will be (batch_size, n_head, seq_len, seq_len), and
        # represent raw attention scores (logits):
        attn_scores = (
            q_heads @                  # (..., seq_len, head_width)
            k_heads.transpose(-2, -1)  # (..., head_width, seq_len)
        ) / math.sqrt(head_width)         # Keep dot prod magnitudes manageable
        # The second-to-last dimension corresponds to the queries of each
        # token, and the last dimension to the keys.
        # attn_scores[15, 7, 29, 18] now represents: "In sequence #15 of the
        # batch, on a scale of -∞ to ∞, how much should head #7 (i.e., the 7th
        # slice of the embedding) of token #29 attend to the 7th slice of token
        # #18's embedding?"
        # Note we're dividing by √head_width here instead of √d_k. That's
        # because in this scheme, d_k effectively *is* head_width.

        # Enforce causal attention. Any attention score from i to j where i < j
        # will get a -∞ score, so that it becomes zero after softmaxing. A
        # token i can only pay attention to tokens 0 through i.
        causal_mask = torch.tril(torch.ones((seq_len, seq_len),
            device=embeddings.device))
        attn_scores.masked_fill_(causal_mask == 0, float("-inf"))

        # attn_weights will also be (batch_size, n_head, seq_len, seq_len)
        # but now the last dimension is on the simplex.
        # So attn_weights[15, 7, 29, 18] represents: "In sequence #15 of the
        # batch, on a scale of 0 to 1, how much should head #7 (i.e., the 7th
        # slice of the embedding) of token #29 attend to the 7th slice of token
        # #18's embedding?"
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # context_vectors_by_head will be (batch_size, n_head, seq_len,
        # head_width). This is, for each head, the weighted sum of all the
        # value vectors for each token in the sequence. (weighted according to
        # Q·K attention scores). (Note: effectively, the "d_v" being used here
        # is d_v / num_head.)
        context_vectors_by_head = attn_weights @ v_heads

        # Finally, now that we've done all the math, put all the heads back
        # together into a single output. This means converting this shape:
        #   (batch_size, n_head, seq_len, head_width)
        # into this shape:
        #   (batch_size, seq_len, d_v)
        context_vectors = self._unsplit_from_heads(context_vectors_by_head)

        # And now, project into this shape:
        #   (batch_size, seq_len, d_model)
        # through a linear output projection.
        mha_outputs = self.output_proj(context_vectors)

        # (If in training mode, randomly zero out some of these to avoid
        # co-adaption of neurons.)
        mha_outputs = self.dropout_layer(mha_outputs)
        return mha_outputs

    def _split_into_heads(
        self,
        tensor: torch.Tensor      # (batch_size, seq_len, d_model)
    ):
        # Given a tensor whose batches are sequences of embeddings, reshape it
        # to be one whose batches are split into heads, each of which operates
        # on smaller, partial embeddings in each sequence. (This way, each head
        # can "attend over the sequence" using a portion of the embedding.)
        batch_size, seq_len, d_model = tensor.size()
        head_width = d_model // self.n_head
        reshaped = tensor.reshape(batch_size, seq_len, self.n_head, head_width)
        return reshaped.transpose(1,2)

    def _unsplit_from_heads(
        self,
        tensor: torch.Tensor      # (batch_size, n_head, seq_len, head_width)
    ):
        # Given a tensor whose batches are split into heads - where each head
        # operates on a smaller, partial embedding of each token in the
        # sequence - reshape it to concatenate all the heads' outputs back into
        # a full embedding.
        batch_size, _, seq_len, head_width = tensor.size()
        d_model = head_width * self.n_head
        transposed = tensor.transpose(1,2)
        # Note: if we had done a .view() here instead of a .reshape(), we'd
        # have a problem because 'transposed' is not contiguous at this point
        # (transposing a tensor breaks contiguity). So we'd have to call
        # .contiguous() immediately after calling .view(). Using .reshape() is
        # safe, because it will automatically copy memory, but only if it's
        # called on a non-contiguous tensor.
        return transposed.reshape(batch_size, seq_len, d_model)
